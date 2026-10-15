"""
Build NDVI composites over fixed-length time windows from PlanetScope scenes.

For every scene, NDVI is computed from the red and near-infrared bands of the
surface reflectance image, and pixels flagged as unusable by the matching UDM2
quality mask are removed. The scenes acquired within each window (in our study,
eight days) are then combined, pixel by pixel, into one composite:

  1. Scenes are found under --input-dir (searched recursively) and assigned a
     date from their file name (PlanetScope scene IDs start with YYYYMMDD_HHMMSS).
  2. The period from --start to --end is split into consecutive windows of
     --window-days days. A final window shorter than --window-days is still
     composited and named with its actual dates.
  3. For each window, every scene is resampled (nearest neighbour) onto one
     common output grid, NDVI = (NIR - red) / (NIR + red) is computed, and the
     UDM2 mask is applied:
       --mask clear          keep only pixels flagged clear (UDM2 band 1)
       --mask cloud shadow   remove pixels flagged in any of the listed bands
                             (cloud, snow, shadow, light_haze, heavy_haze)
       --mask none           no masking
     Optionally, NDVI values above --ndvi-max are removed as well.
  4. The remaining values are combined per pixel with --method:
       median   median of all valid observations (used in our study)
       max      maximum of all valid observations
       sd_max   maximum after removing observations more than --sd-k standard
                deviations from the pixel's mean
  5. The number of valid observations per pixel is written alongside each
     composite.

The output grid is the grid of --reference if given (e.g. a study-area raster),
otherwise the union of all scene footprints in the period, at the resolution and
in the projection of the first scene. All windows share the same grid, so the
composites can be stacked directly (e.g. by backfill_ndvi_windows.py). The data
are processed in tiles, so areas larger than memory can be handled.

This script is shared as an example of the workflow used in our study. The study
data are not distributed; run the script on your own PlanetScope scenes (or any
multispectral scenes with red and NIR bands; use --mask none if they have no
UDM2 mask).

Outputs (in --output-dir)
-------------------------
  <method>_ndvi_window_size_<n>_<start>--<end>_<mask>.tif   composite NDVI
                         (float32, nodata = -9999, LZW compressed, with overviews)
  overlap_window_size_<n>_<start>--<end>_<mask>.tif         number of valid
                         observations per pixel (uint16, 0 = no observations)

Example
-------
python ndvi_window_composites.py --input-dir planet_scenes --output-dir ndvi_8day \
    --start 2020-01-01 --end 2021-01-01 --window-days 8 --method median --mask clear

Requirements: numpy, rasterio.
"""
import argparse
import glob
import math
import os
import re
import warnings
from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

OUT_NODATA = -9999.0
OVERVIEW_FACTORS = [4, 8, 16, 32, 64, 128]
UDM2_BANDS = {'clear': 1, 'snow': 2, 'shadow': 3, 'light_haze': 4, 'heavy_haze': 5, 'cloud': 6}


# --------------------------------------------------------------------------- inputs

def find_scenes(input_dir, scene_suffix, date_regex, date_format):
    """Return a sorted list of (date, scene path) for every scene under input_dir."""
    scenes = []
    for path in glob.glob(os.path.join(input_dir, '**', '*' + scene_suffix), recursive=True):
        match = re.search(date_regex, os.path.basename(path))
        if not match:
            print(f'  warning: no date found in {os.path.basename(path)}; skipped')
            continue
        scenes.append((datetime.strptime(match.group(1), date_format), path))
    return sorted(scenes)


def udm_path(scene_path, scene_suffix, udm_suffix):
    return scene_path[: -len(scene_suffix)] + udm_suffix


def make_windows(start, end, days):
    """Consecutive [start, end) windows of `days` days; the last one may be shorter."""
    windows = []
    w_start = start
    while w_start < end:
        w_end = min(w_start + timedelta(days=days), end)
        windows.append((w_start, w_end))
        w_start = w_end
    return windows


def output_grid(scene_paths, reference):
    """Return (crs, transform, width, height) of the common output grid."""
    if reference:
        with rasterio.open(reference) as ds:
            return ds.crs, ds.transform, ds.width, ds.height
    with rasterio.open(scene_paths[0]) as ds:
        crs, (xres, yres) = ds.crs, ds.res
        x0, y0 = ds.transform.c, ds.transform.f
    left = bottom = math.inf
    right = top = -math.inf
    for p in scene_paths:
        with rasterio.open(p) as ds:
            b = transform_bounds(ds.crs, crs, *ds.bounds) if ds.crs != crs else ds.bounds
        left, bottom = min(left, b[0]), min(bottom, b[1])
        right, top = max(right, b[2]), max(top, b[3])
    # align the grid with the pixels of the first scene
    left = x0 + math.floor(round((left - x0) / xres, 6)) * xres
    top = y0 + math.ceil(round((top - y0) / yres, 6)) * yres
    width = math.ceil(round((right - left) / xres, 6))
    height = math.ceil(round((top - bottom) / yres, 6))
    return crs, from_origin(left, top, xres, yres), width, height


# --------------------------------------------------------------------------- NDVI

class Scene:
    """A scene (and its UDM2 mask) resampled onto the output grid."""

    def __init__(self, path, udm, grid):
        crs, transform, width, height = grid
        vrt_args = dict(crs=crs, transform=transform, width=width, height=height,
                        resampling=Resampling.nearest)
        self._src = rasterio.open(path)
        # areas outside the footprint (and nodata pixels) come back masked
        nodata = self._src.nodata if self._src.nodata is not None else 0
        self.vrt = WarpedVRT(self._src, nodata=nodata, **vrt_args)
        self._udm_src = rasterio.open(udm) if udm else None
        self.udm = WarpedVRT(self._udm_src, nodata=0, **vrt_args) if udm else None

    def ndvi(self, window, red_band, nir_band, mask, ndvi_max):
        bands = self.vrt.read([red_band, nir_band], window=window, masked=True).astype('float32')
        red, nir = bands[0].filled(np.nan), bands[1].filled(np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
        ndvi[~np.isfinite(ndvi)] = np.nan  # nir + red == 0 or missing reflectance

        if mask == ['clear']:
            keep = self.udm.read(UDM2_BANDS['clear'], window=window) == 1
            ndvi[~keep] = np.nan
        elif mask:
            flags = self.udm.read([UDM2_BANDS[m] for m in mask], window=window)
            ndvi[np.any(flags != 0, axis=0)] = np.nan
        if ndvi_max is not None:
            ndvi[ndvi > ndvi_max] = np.nan
        return ndvi

    def close(self):
        for ds in (self.vrt, self._src, self.udm, self._udm_src):
            if ds is not None:
                ds.close()


# --------------------------------------------------------------------------- compositing

def composite(stack, method, sd_k):
    """Combine a (scenes x rows x cols) NDVI stack into one layer, ignoring NaN."""
    with warnings.catch_warnings():
        # pixels with no valid observation give an all-NaN slice; they stay NaN
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if method == 'median':
            return np.nanmedian(stack, axis=0)
        if method == 'max':
            return np.nanmax(stack, axis=0)
        if method == 'sd_max':
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0)
            kept = np.where(np.abs(stack - mean) <= sd_k * sd, stack, np.nan)
            return np.nanmax(kept, axis=0)
    raise ValueError(f'unknown method {method}')


def tiles(width, height, size):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def build_overviews(path):
    with rasterio.Env(COMPRESS_OVERVIEW='DEFLATE'):
        with rasterio.open(path, 'r+') as ds:
            factors = [f for f in OVERVIEW_FACTORS if f <= min(ds.width, ds.height) / 2]
            if factors:
                ds.build_overviews(factors, Resampling.nearest)


def process_window(scenes, grid, out_ndvi, out_count, args, mask):
    crs, transform, width, height = grid
    profile = dict(driver='GTiff', width=width, height=height, count=1, crs=crs,
                   transform=transform, compress='lzw', tiled=True,
                   blockxsize=256, blockysize=256)
    opened = [Scene(p, None if not mask else udm_path(p, args.scene_suffix, args.udm_suffix), grid)
              for p in scenes]
    try:
        with rasterio.open(out_ndvi, 'w', dtype='float32', nodata=OUT_NODATA, **profile) as dst_ndvi, \
                rasterio.open(out_count, 'w', dtype='uint16', **profile) as dst_count:
            for win in tiles(width, height, args.tile_size):
                stack = np.stack([s.ndvi(win, args.red_band, args.nir_band, mask, args.ndvi_max)
                                  for s in opened])
                result = composite(stack, args.method, args.sd_k)
                dst_ndvi.write(np.where(np.isnan(result), OUT_NODATA, result).astype('float32'), 1, window=win)
                dst_count.write(np.count_nonzero(~np.isnan(stack), axis=0).astype('uint16'), 1, window=win)
    finally:
        for s in opened:
            s.close()
    build_overviews(out_ndvi)
    build_overviews(out_count)


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', required=True, help='folder containing the scenes (searched recursively)')
    parser.add_argument('--output-dir', required=True, help='folder for the composites')
    parser.add_argument('--start', required=True, help='first day of the first window (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='day after the last day to include (YYYY-MM-DD)')
    parser.add_argument('--window-days', type=int, default=8, help='window length in days (default: 8)')
    parser.add_argument('--method', choices=['median', 'max', 'sd_max'], default='median',
                        help='how observations are combined per pixel (default: median)')
    parser.add_argument('--sd-k', type=float, default=2.0,
                        help='for sd_max: remove observations more than k standard deviations '
                             'from the mean (default: 2)')
    parser.add_argument('--mask', nargs='+', default=['clear'],
                        choices=['none'] + list(UDM2_BANDS),
                        help="UDM2 masking: 'clear' (default), 'none', or one or more flags to remove "
                             "(cloud snow shadow light_haze heavy_haze)")
    parser.add_argument('--ndvi-max', type=float,
                        help='optional: also remove NDVI values greater than this')
    parser.add_argument('--scene-suffix', default='_AnalyticMS_SR_harmonized_clip.tif',
                        help='file-name ending of the reflectance scenes '
                             '(default: _AnalyticMS_SR_harmonized_clip.tif)')
    parser.add_argument('--udm-suffix', default='_udm2_clip.tif',
                        help='file-name ending of the matching UDM2 masks (default: _udm2_clip.tif)')
    parser.add_argument('--red-band', type=int, default=3,
                        help='red band number (default: 3, 4-band PlanetScope; 6 for 8-band)')
    parser.add_argument('--nir-band', type=int, default=4,
                        help='NIR band number (default: 4, 4-band PlanetScope; 8 for 8-band)')
    parser.add_argument('--date-regex', default=r'(\d{8})_\d{6}',
                        help='regular expression whose first group is the acquisition date in the '
                             r'file name (default: "(\d{8})_\d{6}", PlanetScope scene IDs)')
    parser.add_argument('--date-format', default='%Y%m%d', help='format of that date (default: %%Y%%m%%d)')
    parser.add_argument('--reference', help='optional raster whose grid is used for the outputs')
    parser.add_argument('--tile-size', type=int, default=512, help='processing tile size in pixels (default: 512)')
    args = parser.parse_args()

    if 'none' in args.mask and len(args.mask) > 1 or 'clear' in args.mask and len(args.mask) > 1:
        parser.error("'clear' and 'none' cannot be combined with other masks")
    mask = [] if args.mask == ['none'] else args.mask
    mask_label = 'non_clipped' if not mask else 'clipped_by_' + '_'.join(mask)

    start_time = datetime.now()
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    scenes = [(d, p) for d, p in find_scenes(args.input_dir, args.scene_suffix, args.date_regex, args.date_format)
              if start <= d < end]
    if not scenes:
        raise SystemExit(f'No scenes ending in {args.scene_suffix} between {args.start} and {args.end} '
                         f'in {args.input_dir}')
    if mask:
        missing = [p for _, p in scenes if not os.path.exists(udm_path(p, args.scene_suffix, args.udm_suffix))]
        for p in missing:
            print(f'  warning: no UDM2 mask for {os.path.basename(p)}; scene skipped')
        scenes = [(d, p) for d, p in scenes if p not in missing]

    grid = output_grid([p for _, p in scenes], args.reference)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'{len(scenes)} scenes found; output grid {grid[2]} x {grid[3]} pixels; '
          f'method {args.method}; mask {mask_label}')

    windows = make_windows(start, end, args.window_days)
    for i, (w_start, w_end) in enumerate(windows, 1):
        last_day = w_end - timedelta(days=1)
        in_window = [p for d, p in scenes if w_start <= d < w_end]
        label = f'{w_start:%Y-%m-%d}--{last_day:%Y-%m-%d}'
        if not in_window:
            print(f'[{i}/{len(windows)}] {label}: no scenes; skipped')
            continue
        name = f'window_size_{args.window_days}_{label}_{mask_label}.tif'
        out_ndvi = os.path.join(args.output_dir, f'{args.method}_ndvi_{name}')
        out_count = os.path.join(args.output_dir, f'overlap_{name}')
        process_window(in_window, grid, out_ndvi, out_count, args, mask)
        print(f'[{i}/{len(windows)}] {label}: {len(in_window)} scenes -> {os.path.basename(out_ndvi)}')

    print(f'Finished in {datetime.now() - start_time}')


if __name__ == '__main__':
    main()

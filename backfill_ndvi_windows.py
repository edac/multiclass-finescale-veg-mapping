"""
Back-fill missing pixels in a time series of NDVI composites using neighbouring windows.

Each input raster is a single-band NDVI composite for one time window (in our
study, the median NDVI of clear PlanetScope pixels within an eight-day window).
Removing cloudy and otherwise unusable pixels leaves gaps in some composites.
For each composite, the script fills those gaps as follows:

  1. Take the composites immediately before and after it in the series
     (search distance w = 1) and compute their per-pixel mean, ignoring
     missing values.
  2. Fill each missing pixel of the composite with that mean.
  3. Count the pixels inside the study area that are still missing. If at least
     --missing-threshold pixels remain, widen the search to w = 2 (two composites
     on each side), then w = 3, and so on, always filling the ORIGINAL gaps from
     the mean of all composites within w positions of it.
  4. Write the filled composite once fewer than --missing-threshold pixels remain
     missing, or once the search covers the whole series.

The study area is given by an optional mask raster on the same grid: pixels
equal to 0 are outside the study area and are ignored when counting missing
pixels (they are still filled when neighbouring values exist). Without a mask,
every pixel counts. Pixels equal to a raster's nodata value or NaN are treated
as missing.

The input files are processed in sorted file-name order, so name them so that
sorting gives chronological order (e.g. ndvi_2020-01-01.tif, ndvi_2020-01-09.tif,
...). All rasters must share the same grid (extent, resolution and projection).

This script is shared as an example of the workflow used in our study. The study
data are not distributed; run the script on your own composites.

Outputs (in --output-dir)
-------------------------
  <input name>_backfilled_size_<w>.tif   one per input, where <w> is the search
                                         distance needed (float32, nodata = -9999,
                                         LZW compressed, with overviews)

Example
-------
python backfill_ndvi_windows.py --input-dir ndvi_8day --pattern "median_ndvi_*.tif" \
    --mask study_area_mask.tif --output-dir ndvi_8day_backfilled

Requirements: numpy, rasterio.
"""
import argparse
import glob
import os
import warnings
from datetime import datetime

import numpy as np
import rasterio
from rasterio.enums import Resampling

OUT_NODATA = -9999.0
OVERVIEW_FACTORS = [4, 8, 16, 32, 64, 128]


def read_band(path):
    """Read band 1 as float32, with nodata values converted to NaN."""
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr[arr == ds.nodata] = np.nan
    return arr


def check_grids(paths):
    """Check that all rasters share the same grid, and return the reference profile."""
    with rasterio.open(paths[0]) as ref:
        profile = ref.profile.copy()
        ref_grid = (ref.width, ref.height, ref.transform, ref.crs)
    for p in paths[1:]:
        with rasterio.open(p) as ds:
            if (ds.width, ds.height, ds.transform, ds.crs) != ref_grid:
                raise ValueError(f'{p} is not on the same grid as {paths[0]}')
    return profile


def neighbour_mean(paths, idx, w):
    """Per-pixel mean of all composites within w positions of idx, excluding idx itself."""
    neighbours = [paths[j] for j in range(max(0, idx - w), min(len(paths), idx + w + 1)) if j != idx]
    stack = np.stack([read_band(p) for p in neighbours])
    with warnings.catch_warnings():
        # Pixels missing in every neighbour give an all-NaN slice; they stay NaN.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(stack, axis=0)


def backfill(paths, idx, in_area, missing_threshold):
    """Fill the gaps of composite idx; return the filled array and the search distance used."""
    current = read_band(paths[idx])
    gaps = np.isnan(current)
    max_w = max(idx, len(paths) - 1 - idx)  # distance at which the search covers the whole series

    w = 1
    while True:
        filled = current.copy()
        filled[gaps] = neighbour_mean(paths, idx, w)[gaps]
        n_missing = int(np.count_nonzero(np.isnan(filled) & in_area))
        if n_missing < missing_threshold:
            return filled, w, n_missing
        if w >= max_w:
            print(f'  warning: {n_missing} pixels still missing after searching the whole series')
            return filled, w, n_missing
        w += 1


def write_raster(path, arr, profile):
    """Write a float32 GeoTIFF (NaN -> nodata) with LZW compression and overviews."""
    out = np.where(np.isnan(arr), OUT_NODATA, arr).astype(np.float32)
    profile = profile.copy()
    profile.update(driver='GTiff', count=1, dtype='float32', nodata=OUT_NODATA, compress='lzw')
    with rasterio.Env(COMPRESS_OVERVIEW='DEFLATE'):
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(out, 1)
            dst.build_overviews(OVERVIEW_FACTORS, Resampling.nearest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', required=True, help='folder containing the composite GeoTIFFs')
    parser.add_argument('--pattern', default='*.tif', help='file-name pattern of the composites (default: *.tif)')
    parser.add_argument('--mask', help='optional study-area mask on the same grid (0 = outside the study area)')
    parser.add_argument('--output-dir', required=True, help='folder for the back-filled GeoTIFFs')
    parser.add_argument('--missing-threshold', type=int, default=100,
                        help='stop widening the search once fewer than this many study-area pixels remain '
                             'missing (default: 100; use 1 to require every pixel to be filled)')
    args = parser.parse_args()

    start = datetime.now()
    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if len(paths) < 2:
        raise SystemExit(f'Need at least two composites matching {args.pattern} in {args.input_dir}')
    profile = check_grids(paths)

    if args.mask:
        check_grids([paths[0], args.mask])
        with rasterio.open(args.mask) as ds:
            in_area = ds.read(1) != 0
    else:
        in_area = np.ones((profile['height'], profile['width']), dtype=bool)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'{len(paths)} composites found')

    for idx, path in enumerate(paths):
        filled, w, n_missing = backfill(paths, idx, in_area, args.missing_threshold)
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output_dir, f'{name}_backfilled_size_{w}.tif')
        write_raster(out_path, filled, profile)
        print(f'[{idx + 1}/{len(paths)}] {os.path.basename(path)}: search distance {w}, '
              f'{n_missing} study-area pixels still missing -> {os.path.basename(out_path)}')

    print(f'Finished in {datetime.now() - start}')


if __name__ == '__main__':
    main()

"""
Principal component analysis of a SMAP soil moisture time series with Incremental PCA.

A stack of single-band SMAP soil moisture rasters (one GeoTIFF per date, all on
the same grid) is reduced to principal components that summarise the dominant
spatial and temporal patterns of soil moisture. The data are processed in
windows, so rasters larger than memory can be handled:

  1. Fit: scikit-learn IncrementalPCA is fitted on all valid pixels, window by window.
  2. Transform: each window is projected onto the components and written to one
     GeoTIFF per component.
  3. (Optional) Reconstruction check: the time series is reconstructed from the
     first K components, and the RMSE and R^2 of the reconstruction are reported.

SMAP products are distributed as HDF5; extract the soil moisture layer for each
date to GeoTIFF and bring all dates to a common grid (same extent, resolution and
projection) before running the script. Pixels equal to each raster's nodata value
(e.g. the SMAP fill value -9999) or NaN are treated as missing. The input files
are stacked in sorted file-name order, so name them so that sorting gives
chronological order (e.g. smap_sm_20200101.tif, smap_sm_20200102.tif, ...).

This script is shared as an example of the workflow used in our study. The study
data are not distributed; run the script on your own SMAP soil moisture rasters
(or any single-band time series on a common grid).

Outputs (in --output-dir)
-------------------------
  explained_variance.csv     variance and variance ratio of each component
  pc_01.tif, pc_02.tif, ...  component scores (float32, nodata = -9999)
  reconstruction_check.csv   RMSE and R^2 for K components (only with --check-components)

Example
-------
python smap_timeseries_ipca.py --input-dir smap_soil_moisture --output-dir smap_pca \
    --n-components 10 --check-components 10

Requirements: numpy, rasterio, scikit-learn.
"""
import argparse
import csv
import glob
import os
from datetime import datetime

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.decomposition import IncrementalPCA

OUT_NODATA = -9999.0


def open_stack(paths):
    """Open all rasters and check that they share the same grid."""
    datasets = [rasterio.open(p) for p in paths]
    ref = datasets[0]
    for ds in datasets[1:]:
        if (ds.width, ds.height) != (ref.width, ref.height) or ds.transform != ref.transform or ds.crs != ref.crs:
            raise ValueError(f'{ds.name} is not on the same grid as {ref.name}')
    return datasets


def windows(ds, block):
    for row in range(0, ds.height, block):
        for col in range(0, ds.width, block):
            yield Window(col, row, min(block, ds.width - col), min(block, ds.height - row))


def read_window(datasets, window, fill_nodata_with_zero):
    """Return (pixels x dates) array for the window and a mask of valid pixels."""
    stack = np.stack([ds.read(1, window=window).astype('float64') for ds in datasets])  # dates x h x w
    invalid = ~np.isfinite(stack)
    for k, ds in enumerate(datasets):
        if ds.nodata is not None and np.isfinite(ds.nodata):
            invalid[k] |= stack[k] == ds.nodata
    pixels = stack.reshape(len(datasets), -1).T          # pixels x dates
    bad = invalid.reshape(len(datasets), -1).T
    if fill_nodata_with_zero:
        pixels[bad] = 0.0
        return pixels, np.ones(len(pixels), dtype=bool)
    return pixels, ~bad.any(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--input-dir', required=True, help='Folder with the SMAP soil moisture GeoTIFFs (one per date)')
    parser.add_argument('--pattern', default='*.tif', help='File pattern within --input-dir (default: *.tif)')
    parser.add_argument('--output-dir', required=True, help='Folder for the outputs')
    parser.add_argument('--n-components', type=int, default=None,
                        help='Number of components to write (default: all, i.e. one per input date)')
    parser.add_argument('--block-size', type=int, default=1024, help='Window size in pixels (default: 1024)')
    parser.add_argument('--check-components', type=int, default=None,
                        help='Report reconstruction RMSE and R^2 using the first K components')
    parser.add_argument('--fill-nodata-with-zero', action='store_true',
                        help='Treat nodata pixels as 0 and include them in the fit (default: exclude them)')
    args = parser.parse_args()

    start = datetime.now()
    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if len(paths) < 2:
        raise SystemExit('At least two input rasters are needed.')
    print(f'{len(paths)} input rasters (first: {os.path.basename(paths[0])}, last: {os.path.basename(paths[-1])})')
    datasets = open_stack(paths)
    ref = datasets[0]
    n_dates = len(paths)
    n_keep = min(args.n_components or n_dates, n_dates)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Fit (all components, so the full explained-variance spectrum is available)
    ipca = IncrementalPCA(n_components=n_dates, whiten=False)
    buffer, n_fitted = [], 0
    for window in windows(ref, args.block_size):
        pixels, valid = read_window(datasets, window, args.fill_nodata_with_zero)
        buffer.append(pixels[valid])
        if sum(len(b) for b in buffer) >= max(n_dates, 10000):  # each batch needs >= n_components samples
            batch = np.concatenate(buffer); ipca.partial_fit(batch); n_fitted += len(batch); buffer = []
    if buffer:
        batch = np.concatenate(buffer)
        if len(batch) >= n_dates:
            ipca.partial_fit(batch); n_fitted += len(batch)
    print(f'Fitted on {n_fitted} pixels ({datetime.now() - start})')

    with open(os.path.join(args.output_dir, 'explained_variance.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component', 'explained_variance', 'explained_variance_ratio', 'cumulative_ratio'])
        cumulative = np.cumsum(ipca.explained_variance_ratio_)
        for k in range(n_dates):
            writer.writerow([k + 1, ipca.explained_variance_[k], ipca.explained_variance_ratio_[k], cumulative[k]])

    # 2. Transform and write one GeoTIFF per component
    profile = ref.profile.copy()
    profile.update(driver='GTiff', count=1, dtype='float32', nodata=OUT_NODATA,
                   compress='lzw', tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
    outputs = [rasterio.open(os.path.join(args.output_dir, f'pc_{k + 1:02d}.tif'), 'w', **profile)
               for k in range(n_keep)]
    k_check = min(args.check_components, n_dates) if args.check_components else None
    sse = ss_tot_sum = ss_tot_sq = 0.0
    n_check = 0
    for window in windows(ref, args.block_size):
        pixels, valid = read_window(datasets, window, args.fill_nodata_with_zero)
        scores = np.full((pixels.shape[0], n_keep), OUT_NODATA, dtype='float32')
        if valid.any():
            transformed = ipca.transform(pixels[valid])
            scores[valid] = transformed[:, :n_keep]
            if k_check:
                recon = transformed[:, :k_check] @ ipca.components_[:k_check] + ipca.mean_
                observed = pixels[valid]
                sse += float(((observed - recon) ** 2).sum())
                ss_tot_sum += float(observed.sum()); ss_tot_sq += float((observed ** 2).sum())
                n_check += observed.size
        for k, out in enumerate(outputs):
            out.write(scores[:, k].reshape(int(window.height), int(window.width)), 1, window=window)
    for out in outputs:
        out.close()
    print(f'Wrote {n_keep} component rasters ({datetime.now() - start})')

    if k_check and n_check:
        rmse = (sse / n_check) ** 0.5
        ss_tot = ss_tot_sq - ss_tot_sum ** 2 / n_check
        r2 = 1 - sse / ss_tot if ss_tot > 0 else float('nan')
        with open(os.path.join(args.output_dir, 'reconstruction_check.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['components_used', 'rmse', 'r2'])
            writer.writerow([k_check, rmse, r2])
        print(f'Reconstruction from {k_check} components: RMSE = {rmse:.4f}, R^2 = {r2:.4f}')

    for ds in datasets:
        ds.close()


if __name__ == '__main__':
    main()

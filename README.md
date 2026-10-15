# multiclass-finescale-veg-mapping

This repository contains code examples associated with the publication "Multi-Class Fine-Scale Vegetation Mapping: An Automated Object-Based Workflow for New Mexico Drylands".

The scripts and notebooks are shared as **examples of the workflow** used in the study. The study data contain proprietary information and are **not distributed** with this repository. Every script takes its inputs (file paths, layer and field names, dates, settings) as arguments or settings, so you can run it on your own data, whether your fields are named like ours or differently.

## Workflow

The files are listed in the order they are used:

| Step | File | What it does |
|---|---|---|
| 1 | [`ndvi_window_composites.py`](ndvi_window_composites.py) | Computes NDVI for every PlanetScope scene, removes unusable pixels with the UDM2 quality mask, and combines the scenes in each fixed-length window (eight days in our study) into one composite (per-pixel median by default). It also writes the number of valid observations per pixel. |
| 2 | [`backfill_ndvi_windows.py`](backfill_ndvi_windows.py) | Fills the gaps left in each composite by masked pixels with the mean of the neighbouring composites, widening the search until (almost) every pixel in the study area is filled. |
| 3 | [`smap_timeseries_ipca.py`](smap_timeseries_ipca.py) | Reduces a SMAP soil moisture time series to principal components with Incremental PCA, processing the rasters in windows so areas larger than memory can be handled. It works on any single-band time series on a common grid. |
| 4 | [`assign_plots_to_segments.py`](assign_plots_to_segments.py) | Assigns each field plot to the image segment it overlaps most and flags segments that received several plots or plots with different class labels, so conflicts can be reviewed before training. |
| 5 | [`veg-random-forest-random-split.ipynb`](veg-random-forest-random-split.ipynb) | Trains and evaluates a random forest classifier on segment-level predictors, using stratified random 70/30 training/validation splits. It reports confusion matrices, user's and producer's accuracy, variable importance, and results averaged over many runs, and classifies all segments. |
| 6 | [`veg-rf-shap-class-pairs.ipynb`](veg-rf-shap-class-pairs.ipynb) | Computes class-specific SHAP importance for selected classes, to examine whether classes that are often confused rely on the same predictors. It uses the same splits and seeds as the random-split notebook, so the runs correspond. |

Steps that are not covered here, such as segmenting the imagery and calculating segment-level predictor variables, need to be done with your own tools. The notebooks expect a GeoPackage (or another vector format GeoPandas can read) with one row per segment, a class label column, a column marking the labelled segments, and one column per predictor. See the first cells of each notebook for details.

## Usage

Each script describes its method, inputs and outputs at the top of the file, and lists its options with `--help`:

```
python ndvi_window_composites.py --help
```

For example, to build eight-day median NDVI composites for 2020 and then back-fill them:

```
python ndvi_window_composites.py --input-dir planet_scenes --output-dir ndvi_8day \
    --start 2020-01-01 --end 2021-01-01 --window-days 8 --method median --mask clear

python backfill_ndvi_windows.py --input-dir ndvi_8day --pattern "median_ndvi_*.tif" \
    --mask study_area_mask.tif --output-dir ndvi_8day_backfilled
```

Only complete windows are composited, so a year of eight-day windows gives 45 composites; the remaining days at the end of the period are not used.

The notebooks are run cell by cell in Jupyter. Set the path to your data file and the field names in their settings cells.

## Requirements

Python 3.10 or later, with:

* `numpy`, `pandas`
* `rasterio` (raster scripts)
* `scikit-learn` (PCA and random forest)
* `geopandas` with `pyogrio` or `fiona`, and `shapely` (vector data)
* `matplotlib`, `seaborn`, `joblib` (random forest notebook)
* `shap` (SHAP notebook)

For example:

```
pip install numpy pandas rasterio scikit-learn geopandas pyogrio shapely matplotlib seaborn joblib shap jupyter
```

## License

Released under the [MIT License](LICENSE).

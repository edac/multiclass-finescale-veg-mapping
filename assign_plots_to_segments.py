"""
Assign field plots to image segments and flag conflicting assignments.

Each training plot (polygon) is assigned to the single image segment with which
it shares the largest area of overlap. The script then identifies segments that
received more than one plot and, of those, segments whose plots carry different
class labels. These conflicts were reviewed (and the affected plots excluded)
before the plots were used for random forest training and validation.

Inputs
------
A GeoPackage (or any vector format readable by GeoPandas) containing:
  * a plot layer with a unique plot identifier and a class label field, and
  * a segment layer with a unique segment identifier.

Outputs (CSV, written to --output-dir)
--------------------------------------
  plot_segment_assignment.csv     one row per plot: assigned segment, overlap area
                                  and the fraction of the plot covered by that segment
  segments_with_multiple_plots.csv  segments that received more than one plot
  segments_with_class_conflicts.csv segments whose plots have different class labels

Example
-------
python assign_plots_to_segments.py \
    --gpkg project.gpkg --plots-layer training_plots --segments-layer segments \
    --plot-id PlotID --segment-id SegmentID --class-field Class \
    --output-dir plot_assignment

Requirements: geopandas (>= 0.14, with pyogrio or fiona), pandas, shapely.
"""
import argparse
import os

import geopandas as gpd
import pandas as pd


def read_segments_near_plots(path, layer, plots):
    """Read only the segments that intersect the plots (segment layers can be very large)."""
    try:
        mask = plots.geometry.union_all()
    except AttributeError:  # geopandas < 1.0
        mask = plots.geometry.unary_union
    return gpd.read_file(path, layer=layer, mask=mask)


def assign_plots(plots, segments, plot_id, segment_id, class_field):
    """Return one row per plot with the segment of largest overlap."""
    if plots.crs != segments.crs:
        plots = plots.to_crs(segments.crs)

    plots = plots[[plot_id, class_field, 'geometry']].copy()
    segments = segments[[segment_id, 'geometry']].copy()

    # Candidate plot-segment pairs, then the actual overlap area of each pair
    candidates = gpd.sjoin(plots, segments, how='inner', predicate='intersects')
    segment_geom = segments.set_index(segment_id).geometry
    candidates['overlap_area'] = [
        plot_geom.intersection(segment_geom.loc[seg]).area
        for plot_geom, seg in zip(candidates.geometry, candidates[segment_id])
    ]
    candidates['plot_area'] = candidates.geometry.area

    # Keep the segment with the largest overlap for each plot (first one on ties)
    best = (candidates.sort_values('overlap_area', ascending=False, kind='stable')
                      .drop_duplicates(subset=plot_id, keep='first'))
    best['overlap_fraction'] = best['overlap_area'] / best['plot_area']
    out = best[[plot_id, class_field, segment_id, 'overlap_area', 'overlap_fraction']].sort_values(plot_id)
    return pd.DataFrame(out)


def summarise_segments(assignment, plot_id, segment_id, class_field):
    """Segments receiving more than one plot, and the subset with conflicting classes."""
    grouped = assignment.groupby(segment_id).agg(
        n_plots=(plot_id, 'nunique'),
        plot_ids=(plot_id, lambda s: ';'.join(map(str, sorted(s.unique())))),
        n_classes=(class_field, 'nunique'),
        classes=(class_field, lambda s: ';'.join(map(str, sorted(s.unique())))),
    ).reset_index()
    multiple = grouped[grouped['n_plots'] > 1]
    conflicts = multiple[multiple['n_classes'] > 1]
    return multiple, conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--gpkg', required=True, help='Path to the GeoPackage (or other vector file)')
    parser.add_argument('--plots-layer', required=True, help='Layer containing the training plots')
    parser.add_argument('--segments-layer', required=True, help='Layer containing the image segments')
    parser.add_argument('--plot-id', required=True, help='Unique plot identifier field')
    parser.add_argument('--segment-id', required=True, help='Unique segment identifier field')
    parser.add_argument('--class-field', required=True, help='Plot class label field')
    parser.add_argument('--output-dir', default='plot_assignment', help='Folder for the output CSV files')
    args = parser.parse_args()

    plots = gpd.read_file(args.gpkg, layer=args.plots_layer)
    segments = read_segments_near_plots(args.gpkg, args.segments_layer, plots)
    print(f'{len(plots)} plots; {len(segments)} segments intersect the plots')

    assignment = assign_plots(plots, segments, args.plot_id, args.segment_id, args.class_field)
    multiple, conflicts = summarise_segments(assignment, args.plot_id, args.segment_id, args.class_field)

    os.makedirs(args.output_dir, exist_ok=True)
    assignment.to_csv(os.path.join(args.output_dir, 'plot_segment_assignment.csv'), index=False)
    multiple.to_csv(os.path.join(args.output_dir, 'segments_with_multiple_plots.csv'), index=False)
    conflicts.to_csv(os.path.join(args.output_dir, 'segments_with_class_conflicts.csv'), index=False)

    print(f'Plots assigned to a segment: {len(assignment)} of {len(plots)}')
    print(f'Segments with more than one plot: {len(multiple)}')
    print(f'Segments with plots of different classes: {len(conflicts)}')
    print(f'Results written to {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()

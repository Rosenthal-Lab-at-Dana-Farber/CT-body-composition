import os
import json

import argparse
from pathlib import Path

from body_comp.inference.components.preprocess import (
    FindSeries,
    WindowAndShiftDICOMSeries,
    FillMetadata,
)
from body_comp.inference.components.filter import (
    CheckCircleCrop,
    CheckDICOMDecompressionError,
    CheckEntangledSeries,
    CheckModality,
    CheckOrientation,
    CheckPixelArraySize,
    CheckSeriesLength,
    CheckType,

)
from body_comp.inference.pipeline import Pipeline
from body_comp.inference.components.slice_selector import SliceSelector
from body_comp.inference.components.body_comp_estimator import BodyCompositionEstimator
from body_comp.inference.utils import (
    filter_csv,
    summarize_qc,
    write_qc_to_csv,
    write_results_to_csv,
)


def main(args):

    num_threads = args.num_threads if args.num_threads else os.cpu_count()

    estimator_config = {}
    slice_selector_config = {}

    # Need to decouple slice selection specific parameters from body comp estimator specific ones
    if isinstance(args.estimator_config, (str, Path)):
        with open(str(args.estimator_config), "r") as jsonfile:
            estimator_config = json.load(jsonfile)
            if "slice_selection_weights" in estimator_config:
                slice_selector_config["slice_selection_weights"] = estimator_config.pop(
                    "slice_selection_weights"
                )
            if "sigmoid_output" in estimator_config:
                slice_selector_config["sigmoid_output"] = estimator_config[
                    "sigmoid_output"
                ]

    transforms = [
        FindSeries(),
        FillMetadata(),
        CheckEntangledSeries(),
        CheckType(),
        CheckModality(),
        CheckSeriesLength(min_series_length=args.min_slices_per_series),
        CheckOrientation(),
        CheckCircleCrop(),
        CheckPixelArraySize(),
        CheckDICOMDecompressionError(),
        WindowAndShiftDICOMSeries(),
        SliceSelector(
            **slice_selector_config,
            num_threads=num_threads,
        ),
        # if not using slice selection in pipeline, "slice_selection" needs to be set to False
        BodyCompositionEstimator(
            **estimator_config,
            num_threads=num_threads,
            output_dir=args.output_dir,
            segmentation_range=args.segmentation_range,
            keep_existing=args.keep_existing,
            dicom_seg=args.dicom_seg,
            slice_selection=args.slice_selection,
        ),
    ]

    pipeline = Pipeline(transforms)

    pipeline.apply(args.root, series=args.slice_selection)

    pipeline.save_cohort_summary(os.path.join(args.output_dir, "results.json"))

    write_qc_to_csv(
        pipeline.cohort_summary, os.path.join(args.output_dir, "qc_results.csv")
    )

    summarize_qc(
        os.path.join(args.output_dir, "qc_results.csv"),
        os.path.join(args.output_dir, "qc_summary.csv"),
    )

    multislice = True if args.segmentation_range else False

    # if not using slice selection in above pipeline, "slice_selection" needs to be set to False
    write_results_to_csv(
        pipeline.cohort_summary,
        os.path.join(args.output_dir, "results.csv"),
        slice_selection=args.slice_selection,
        multislice=multislice,
    )

    if args.slice_selection:
        filter_csv(
            os.path.join(args.output_dir, "results.csv"),
            os.path.join(args.output_dir, "filtered_results.csv"),
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the body composition algorithm on a cohort of patients in a csv file"
    )
    parser.add_argument("root", help="Directory containing input files")
    parser.add_argument(
        "output_dir",
        help=(
            "Output directory for image results. Should be empty unless "
            "keep_existing is specified."
        ),
    )
    parser.add_argument(
        "--estimator_config",
        "-c",
        help="A json file containing parameters for the estimator",
    )
    parser.add_argument(
        "--num_threads", "-t", type=int, help="Number of threads to use"
    )
    parser.add_argument(
        "--segmentation_range",
        "-r",
        type=int,
        help=(
            "Segment all slices with this distance (in mm) of the selected slice. "
            "Leave unspecified for single slice."
        ),
    )
    parser.add_argument(
        "--keep_existing",
        "-k",
        action="store_true",
        help="Skip studies that already have a file in the output directory",
    )
    parser.add_argument(
        "--dicom_seg", "-d", action="store_true", help="Save dicom seg files"
    )
    parser.add_argument(
        "--min_slices_per_series",
        "-m",
        type=int,
        default=20,
        help="Reject series with fewer than this number of instances",
    )
    parser.add_argument(
        "--no_slice_selection",
        "-s",
        action="store_false",
        dest="slice_selection",
        help=(
            "Do not perform slice selection as part of analysis pipeline. "
            "By default, slice selection is performed."
        ),
    )
    args = parser.parse_args()

    main(args)

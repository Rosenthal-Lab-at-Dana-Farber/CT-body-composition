import os
import json

import argparse
from pathlib import Path
import importlib.resources

import body_comp
from body_comp.inference.components.preprocess import (
    FindSeries,
    FindSlices,
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

    if not args.slice_selection and args.segmentation_range is not None:
        raise ValueError(
            "Specifying a segmentation range is valid only when "
            "slice selection is performed."
        )

    num_threads = args.num_threads if args.num_threads else os.cpu_count()

    estimator_config = {}
    slice_selector_config = {}

    # Need to decouple slice selection specific parameters from body comp estimator specific ones
    if args.estimator_config is None:
        estimator_config_path = (
            importlib.resources.files(body_comp) /
            "configs/default_l3_config.json"
        )
    else:
        estimator_config_path = args.estimator_config
    with open(str(estimator_config_path), "r") as jsonfile:
        estimator_config = json.load(jsonfile)
        if "slice_selection_weights" in estimator_config:
            slice_selector_config["slice_selection_weights"] = estimator_config.pop(
                "slice_selection_weights"
            )
        if "sigmoid_output" in estimator_config:
            slice_selector_config["sigmoid_output"] = estimator_config[
                "sigmoid_output"
            ]

    transforms = []

    if args.slice_selection:
        transforms.append(FindSeries())
    else:
        transforms.append(FindSlices())

    transforms.append(FillMetadata())

    if args.slice_selection:
        transforms.extend(
            [
                CheckEntangledSeries(),
                CheckSeriesLength(min_series_length=args.min_slices_per_series),
            ]
        )

    transforms.extend(
        [
            CheckType(),
            CheckModality(),
            CheckOrientation(),
            CheckCircleCrop(),
            CheckPixelArraySize(),
            CheckDICOMDecompressionError(),
            WindowAndShiftDICOMSeries(),
        ]
    )

    if args.slice_selection:
        transforms.append(
            SliceSelector(
                **slice_selector_config,
                num_threads=num_threads,
            )
        )

    transforms.append(
        BodyCompositionEstimator(
            **estimator_config,
            num_threads=num_threads,
            output_dir=args.output_dir,
            segmentation_range=args.segmentation_range,
            keep_existing=args.keep_existing,
            dicom_seg=args.dicom_seg,
            slice_selection=args.slice_selection,
        )
    )

    pipeline = Pipeline(transforms)

    summary_path = os.path.join(args.output_dir, "results.json")
    pipeline.apply(
        args.root,
        series=args.slice_selection,
        summary_path=summary_path,
        study_depth=args.study_depth,
        study_list=args.study_list,
        keep_existing=args.keep_existing,
    )

    pipeline.save_cohort_summary(summary_path)

    if len(pipeline.cohort_summary) == 0:
        return

    write_qc_to_csv(
        pipeline.cohort_summary, os.path.join(args.output_dir, "qc_results.csv")
    )

    summarize_qc(
        os.path.join(args.output_dir, "qc_results.csv"),
        os.path.join(args.output_dir, "qc_summary.csv"),
    )

    multislice = args.segmentation_range is not None

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
        description="Run the body composition algorithm on a cohort of patients in a directory"
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
            "Leave unspecified for single slice. Note that this option requires "
            "that slice selection is *not* disabled with the --no_slice_selection "
            "option."
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
        "--study_depth",
        "-e",
        type=int,
        help=(
            "If not provided, every subdirectory at any level of the hierarchy under "
            "'root' that contains files is considered a study. If study_depth is "
            "a non-negative integer, then each directory that number of levels "
            "below root is considered a study. E.g. if study_depth is 0, the "
            "root directory itself is a single study. If study_depth is 1, each "
            "sub-directory of root is considered a study. If study_depth is 2, "
            "each sub-directory of a sub-directory of root is considered a "
            "study, etc. If this option is used, any file at any level under a "
            "study directory is included (for example, files may be grouped into "
            "series directories under the study level). Incompatible with "
            "study_list."
        ),
    )
    parser.add_argument(
        "--study_list",
        "-l",
        help=(
            "Path to a file containing a list of studies to process. This "
            "allows you to specify that only a subset of studies under the "
            "root be processed (by default all are processed). The file "
            "should be a plain text. Each line within the file should contain "
            "the path to a single study to be processed, given relative to "
            "the root directory. Incompatible with study_depth."
        ),
    )
    parser.add_argument(
        "--no_slice_selection",
        "-s",
        action="store_false",
        dest="slice_selection",
        help=(
            "Do not perform slice selection as part of analysis pipeline and "
            "instead run the segmentation on every slice. Note that model "
            "accuracy deteriorates away from the slices it was trained on "
            "(L3 for the default model). By default, slice selection is "
            "performed and the segmentation is performed only on the selected "
            "slices."
        ),
    )
    args = parser.parse_args()

    main(args)

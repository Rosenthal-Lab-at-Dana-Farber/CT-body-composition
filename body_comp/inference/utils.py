import multiprocessing as mp
import os
import functools
import numpy as np
import shutil

import pandas as pd

import pydicom

from .configs import STUDY_LEVEL_TAGS, SERIES_LEVEL_TAGS, INSTANCE_LEVEL_TAGS


def is_uid(col_name: str) -> bool:
    """Determine whether a column name from a CSV file represents a UID

    Parameters
    ----------
    col_name: str
        Name of the column

    Returns
    -------
    bool
        True if and only if the column name represents a uid

    """
    if col_name in ("series_instance_uid", "study_instance_uid"):
        return True
    if "sop_instance_uid" in col_name:
        return True
    return False


def read_file(filepath, list_tags=None, stop_before_pixels=False):

    """Read DICOM metadata from a given filepath.

    Parameters
    ----------
    filespath: str
        File path to read DICOM metadata from.
    list_tags: list
        If used, only the supplied tags will be returned. The supplied elements can be tags or keywords.
    stop_before_pixels: bool
        If False, the full file will be read and parsed. Set True to stop before reading (7FE0,0010) Pixel Data (and
        all subsequent elements).

    Returns
    -------
    pydicom DICOM object
        pydicom Dataset object read from the corresponding DICOM file.

    """

    try:
        meta = pydicom.filereader.read_file(
            filepath, stop_before_pixels=stop_before_pixels, specific_tags=list_tags
        )
        return meta
    except (pydicom.errors.InvalidDicomError, IsADirectoryError):
        return None
    except OSError:
        return None


def read_files_list(
    files_list, num_threads=32, list_tags=None, stop_before_pixels=False
):

    """Read DICOM metadata from a list of filepaths and append to array.

    Parameters
    ----------
    files_list: list
        List of DICOM file names.
    num_threads: int
        Number of processes to create (using python multiprocessing) to read in image files.
    list_tags: list
        If used, only the supplied tags will be returned. The supplied elements can be tags or keywords.
    stop_before_pixels: bool
        If False, the full file will be read and parsed. Set True to stop before reading (7FE0,0010) Pixel Data (and
        all subsequent elements).

    Returns:
    --------
    list
        List of pydicom Dataset objects read from the corresponding DICOM files list.

    """
    # Read in list of files with multithreading
    if num_threads > 1:
        pool = mp.Pool(num_threads)
        func = functools.partial(
            read_file, list_tags=list_tags, stop_before_pixels=stop_before_pixels
        )
        results = pool.map(func, files_list)
        pool.close()
    else:
        results = [
            read_file(f, list_tags=list_tags, stop_before_pixels=stop_before_pixels)
            for f in files_list
        ]

    results = [dcm for dcm in results if dcm is not None]
    return results


# utility to allow dict containing numpy arrays to be written to JSON file
def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Not serializable")


# Get DICOM metadata in a dictionary and apply normalization
def get_dicom_metadata(dcm, tags_dict, results=None):

    if results is None:
        results = {}

    for key, value in tags_dict.items():
        if value["keyword"] in dcm:
            tag_val = getattr(dcm, value["keyword"])
            try:
                results[key] = value["type"](tag_val)
            except (ValueError, TypeError):
                results[key] = None
        else:
            results[key] = None

    # perform custom check for series description tag if present
    if "series_description" in list(tags_dict.keys()):
        # Occasionally series descriptions are missing, or in very odd situations, are lists
        series_description = (
            dcm.SeriesDescription
            if "SeriesDescription" in dcm
            else None
        )
        if isinstance(series_description, pydicom.multival.MultiValue):
            series_description = " ".join(list(series_description))
        results["series_description"] = series_description

    return results


# Windowing function
def apply_window(image, win_centre, win_width):
    range_bottom = win_centre - win_width / 2
    scale = 256 / float(win_width)
    image = image - range_bottom

    image = image * scale

    image[image < 0] = 0
    image[image > 255] = 255

    return image


# Apply rescale shift
def rescale_shift(image, intercept, slope):
    return image * slope + intercept


### Functions for writing results to csv files


def write_qc_to_csv(results, output_file):

    output_list = []

    for study_path, study_summary in results.items():

        for series_uid, series_data in study_summary["UIDs"].items():

            series_results = {"study_path": study_path}
            series_results["study_name"] = study_summary["study_name"]
            series_results["series_instance_uid"] = series_uid

            # Study level information
            for key in STUDY_LEVEL_TAGS:
                if key in study_summary:
                    series_results[key] = study_summary[key]

            # Series level information
            for key in SERIES_LEVEL_TAGS:
                if key in series_data:
                    series_results[key] = series_data[key]

            series_results["valid_series"] = series_data["valid"]
            series_results["reason_for_disqualification"] = series_data["reason"]

            output_list.append(series_results)

    pd.DataFrame(output_list).to_csv(output_file)


def write_results_to_csv(
    results, output_file, multislice=False, center=False, slice_selection=False
):

    output_list = []

    for study_path, study_summary in results.items():

        num_valid_series = sum(
            [series_data["valid"] for _, series_data in study_summary["UIDs"].items()]
        )

        for series_uid, series_data in study_summary["UIDs"].items():

            if not series_data["valid"]:
                continue

            series_results = {"study_path": study_path}
            series_results["study_name"] = study_summary["study_name"]

            series_results["series_instance_uid"] = series_uid
            series_results["num_images"] = len(series_data["files"])

            series_results["num_valid_series"] = num_valid_series

            # Study level information
            for key in STUDY_LEVEL_TAGS:
                if key in study_summary:
                    series_results[key] = study_summary[key]

            # Series level information
            for key in SERIES_LEVEL_TAGS:
                if key in series_data:
                    series_results[key] = series_data[key]

            if slice_selection:

                for slice_name, slice_data in series_data["slice selection"]["results"][
                    "slices"
                ].items():

                    series_results["slice_index"] = series_data["slice selection"][
                        "results"
                    ]["slices"][slice_name]["index"]
                    series_results[
                        "{}_zero_crossings".format(slice_name)
                    ] = series_data["slice selection"]["results"]["slices"][slice_name][
                        "num_zero_crossings"
                    ]
                    series_results[
                        "{}_regression_val".format(slice_name)
                    ] = series_data["slice selection"]["results"]["slices"][slice_name][
                        "regression_val"
                    ]

            for slice_name, slice_data in series_data["body composition"]["results"][
                "slices"
            ].items():

                # Where to find the data to extract to the CSV now depends on whether multislice analysis was run
                if multislice:
                    # Use only the metadata and, if specified, results from the center slice
                    for s in slice_data["individual"]:
                        if s["offset_from_chosen"] == 0.0:
                            if center:
                                tissue_items = s["tissues"].items()
                            # Copy over instance-level metadata
                            for key in INSTANCE_LEVEL_TAGS:
                                if key in slice_data:
                                    series_results[
                                        "{}_{}".format(slice_name, key)
                                    ] = slice_data[key]

                            series_results[
                                "{}_sop_instance_uid".format(slice_name)
                            ] = slice_data["sop_instance_uid"]
                            series_results[
                                "{}_z_location".format(slice_name)
                            ] = slice_data["z_location"]
                            break
                    else:
                        print(
                            "Warning: no center slice found for {} in study {} series {}!".format(
                                slice_name, series_results["study_name"], series_uid
                            )
                        )
                        continue
                    if not center:
                        # Default for multislice anlysis is to use the overall values aggregated from all the slices
                        tissue_items = slice_data["overall"]["tissues"].items()
                else:
                    # Copy over instance-level metadata
                    for key in INSTANCE_LEVEL_TAGS:
                        if key in slice_data:
                            series_results[
                                "{}_{}".format(slice_name, key)
                            ] = slice_data[key]

                    series_results[
                        "{}_sop_instance_uid".format(slice_name)
                    ] = slice_data["sop_instance_uid"]
                    series_results["{}_z_location".format(slice_name)] = slice_data[
                        "z_location"
                    ]

                    # Simple single slice anaysis - use results from the single slice
                    tissue_items = slice_data["tissues"].items()

                for tissue_name, tissue_data in tissue_items:

                    for prop in [
                        "median_hu",
                        "std_hu",
                        "mean_hu",
                        "area_cm2",
                        "iqr_hu",
                        "boundary_check",
                    ]:
                        series_results[
                            "{}_{}_{}".format(slice_name, tissue_name, prop)
                        ] = tissue_data[prop]

            output_list.append(series_results)

    pd.DataFrame(output_list).to_csv(output_file)


def filter_csv(
    input_csv,
    output_csv,
    choose_thickest=True,
    slices=["L3"],
    boundary_checks=False,
    output_dir=None,
):

    # Open the input csv with no rows to get the columns names
    df = pd.read_csv(str(input_csv), index_col=0, nrows=0)

    # Ensure any column containing a UID is read in as a string
    column_types = {c: str for c in df.columns if is_uid(c)}

    # Open the input csv
    df = pd.read_csv(str(input_csv), index_col=0, dtype=column_types)
    initial_len = len(df)
    initial_n_studies = df.study_path.nunique()
    print("initial studies", initial_n_studies)

    # Initially include all series, then filter out unwanted ones
    ind = pd.Series(True, index=df.index)

    for slice_name in slices:
        # Filter series with multiple zero-crossings or no zero-crossings
        ind &= df["{}_zero_crossings".format(slice_name)] == 1

        # Check how many studies were rejected based on slice selection
        n_studies_after_slice_selection = df[ind].study_path.nunique()
        print(
            initial_n_studies - n_studies_after_slice_selection,
            "studies dropped due to slice selection",
        )

        if boundary_checks:
            # Filter based on boundary checks
            for tis, val in BOUNDARY_CHECKS.items():
                ind &= df["{}_{}_boundary_check".format(slice_name, tis)] <= val

            # Check number of studies after boundary checks
            n_studies_after_boundary_check = df[ind].study_name.nunique()
            print(
                n_studies_after_slice_selection - n_studies_after_boundary_check,
                "further dropped after boundary checks",
            )

    # Apply the index
    df = df[ind].copy()

    if choose_thickest:
        # Sort by slice thickness within a given study and then drop thinner sliced studies, smaller series and
        # slices with lower visceral fat amounts at L3
        df = (
            df.sort_values(
                by=[
                    "study_path",
                    "slice_thickness_mm",
                    "num_images",
                    "L3_visceral_fat_area_cm2",
                ]
            )
            .drop_duplicates(subset=["study_path"], keep="last")
            .copy()
        )

    # copy all QA files that passed into filtered summary into 'ready_for_qa' folder
    studies = list(df["study_name"].to_numpy())
    series_uid = list(df["series_instance_uid"].to_numpy())
    for study, series in zip(studies, series_uid):
        path = os.path.join(
            output_dir, "previews", "{}_{}_preview.png".format(study, series)
        )
        new_path = os.path.join(
            output_dir, "ready_for_qa", "{}_{}_preview.png".format(study, series)
        )
        shutil.copy(path, new_path)

    # Store to file
    print("{} of {} initial series retained".format(len(df), initial_len))
    df.to_csv(str(output_csv))


def summarize_qc(qc_file, output_file):
    df = pd.read_csv(qc_file)
    reason_for_disqualification = df["reason_for_disqualification"].to_numpy()
    reason, count = np.unique(reason_for_disqualification, return_counts=True)
    data = np.concatenate(
        [np.expand_dims(reason, axis=1), np.expand_dims(count, axis=1)], axis=1
    )
    df = pd.DataFrame(
        data, columns=["reason for disqualification", "number of scans disqualified"]
    )
    df.to_csv(output_file)

import os

import numpy as np

from ..utils import (
    apply_window,
    get_dicom_metadata,
    read_files_list,
    rescale_shift,
)
from ..configs import SERIES_LEVEL_TAGS, STUDY_LEVEL_TAGS
from .component import Component, SeriesSummary, StudySummary


class FindSeries(Component):

    def __init__(self, num_threads=None, stop_before_pixels=False):
        """

        Component object that identifies all series in a batch of files and
        creates a dictionary mapping filenames/pydicom datasets to their
        respective SeriesUID tags.

        Parameters:
        -----------
        num_threads: int
            Number of processes to create (using python multiprocessing) to
            read in image files.
        stop_before_pixels: bool
            If False, the full file will be read and parsed. Set True to stop
            before reading (7FE0,0010) Pixel Data (and all subsequent
            elements).
        """
        super().__init__()

        if num_threads is None:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                cpu_count = 1
            self.num_threads = cpu_count
        else:
            self.num_threads = num_threads
        self.stop_before_pixels = stop_before_pixels

    def F(self, study_summary: StudySummary):

        datasets = read_files_list(
            study_summary["files"],
            num_threads=self.num_threads,
            stop_before_pixels=self.stop_before_pixels,
        )

        study_name = f"{datasets[0].PatientID}_{datasets[0].AccessionNumber}"
        study_summary["study_name"] = study_name

        study_uid = datasets[0].StudyInstanceUID

        for file, dcm in zip(study_summary["files"], datasets):
            if dcm.StudyInstanceUID != study_uid:
                raise RuntimeError(
                    f"Found mismatched study instance UIDs within {study_name}"
                )
            study_summary["UIDs"][dcm.SeriesInstanceUID]["files"].append(file)
            study_summary["UIDs"][dcm.SeriesInstanceUID]["datasets"].append(dcm)
            study_summary["UIDs"][dcm.SeriesInstanceUID][
                "PrimaryUID"
            ] = dcm.SeriesInstanceUID

            study_summary["UIDs"][dcm.SeriesInstanceUID]["droppable_keys"].append(
                "datasets"
            )

        return study_summary

    def apply(self, study_summary):

        study_summary = self.F(study_summary)

        study_summary.pop("files")

        return study_summary


class FindSlices(Component):
    def __init__(self, num_threads=None, stop_before_pixels=False):
        """

        Component that creates a study summary out of every individual dicom
        file in a folder, for when single slice analysis is needed instead of
        series level analysis.

        Parameters
        ----------
        num_threads: int
            Number of processes to create (using python multiprocessing) to
            read in image files.
        stop_before_pixels: bool
            If False, the full file will be read and parsed. Set True to stop
            before reading (7FE0,0010) Pixel Data (and all subsequent
            elements).
        """
        super().__init__()

        if num_threads is None:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                cpu_count = 1
            self.num_threads = cpu_count
        else:
            self.num_threads = num_threads
        self.stop_before_pixels = stop_before_pixels

    def F(self, study_summary: StudySummary):

        datasets = read_files_list(
            study_summary["files"],
            num_threads=self.num_threads,
            stop_before_pixels=self.stop_before_pixels,
        )

        study_name = f"{datasets[0].PatientID}_{datasets[0].AccessionNumber}"
        study_summary["study_name"] = study_name
        study_uid = datasets[0].StudyInstanceUID

        for file, dcm in zip(study_summary["files"], datasets):
            if dcm.StudyInstanceUID != study_uid:
                raise RuntimeError(
                    f"Found mismatched study instance UIDs within {study_name}"
                )
            study_summary["UIDs"][dcm.SOPInstanceUID]["Path"] = file
            study_summary["UIDs"][dcm.SOPInstanceUID]["datasets"].append(dcm)
            study_summary["UIDs"][dcm.SOPInstanceUID][
                "PrimaryUID"
            ] = dcm.SOPInstanceUID

            study_summary["UIDs"][dcm.SOPInstanceUID]["droppable_keys"].append(
                "datasets"
            )

        return study_summary

    def apply(self, study_summary: StudySummary):

        study_summary = self.F(study_summary)

        study_summary.pop("files")

        return study_summary


class FillMetadata(Component):
    def __init__(
        self, series_level_tags=SERIES_LEVEL_TAGS, study_level_tags=STUDY_LEVEL_TAGS
    ):
        """

        Component object that pulls out requested metadata at the study and series level.

        """
        super().__init__()

        self.series_level_tags = series_level_tags
        self.study_level_tags = study_level_tags

    def F(self, series_summary: SeriesSummary):

        series_summary = get_dicom_metadata(
            series_summary["datasets"][0],
            self.series_level_tags,
            results=series_summary,
        )

        return series_summary

    def apply(self, study_summary):

        for series_uid, series_summary in study_summary["UIDs"].items():
            study_summary["UIDs"][series_uid] = self.F(series_summary)

        study_summary = get_dicom_metadata(
            series_summary["datasets"][0], self.study_level_tags, results=study_summary
        )

        return study_summary


class WindowAndShiftDICOMSeries(Component):

    def __init__(self, win_centre=40, win_width=400):
        super().__init__()
        self.win_centre = win_centre
        self.win_width = win_width

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        images_list = []
        windowed_images = []
        z_locations = []

        # Sort the datasets by physical location in the z direction
        series_summary["datasets"] = sorted(
            series_summary["datasets"],
            key=lambda dcm: float(dcm.ImagePositionPatient[2]),
        )

        for dcm in series_summary["datasets"]:

            image = dcm.pixel_array

            slope = float(dcm.RescaleSlope)
            intercept = float(dcm.RescaleIntercept)
            image = rescale_shift(image, intercept, slope)

            # Rotate image to supine position if not in supine
            patient_position = dcm.PatientPosition
            if patient_position.endswith("P"):  # prone
                image = np.rot90(image, k=2)
            elif patient_position.endswith("DL"):  # decubitus left
                image = np.rot90(image, k=1)
            elif patient_position.endswith("DR"):  # decubitus right
                image = np.rot90(image, k=3)

            images_list.append(image)

            # Windowing preprocessing
            windowed_images.append(apply_window(image, self.win_centre, self.win_width))

            z_locations.append(float(dcm.ImagePositionPatient[2]))

        series_summary["images"] = images_list
        series_summary["windowed_images"] = windowed_images
        series_summary["z_locations"] = z_locations

        series_summary["droppable_keys"].extend(
            ["images", "windowed_images", "z_locations"]
        )

        return series_summary

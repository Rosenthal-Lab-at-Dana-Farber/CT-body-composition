from collections import defaultdict

import numpy as np

from .component import Component, SeriesSummary, StudySummary


_ERROR_MESSAGES = {
    "missing tags": "Series is missing tag(s) required for QC, cannot perform QC",
    "seriesLength": "Series does not meet the minimum length requirements for analysis.",
    "modality": "Series modality is not CT",
    "imageType": {
        "DERIVED": "Series contains derived images",
        "GSI MD": "Series contains dual-energy images that are not HU-based",
    },
    "circleCrop": "There is a high likelihood that this series has been cropped, requires manual inspection",
    "entangledSeries": "Multiple series stacked together",
    "missingImagePositionPatient": "Slice in series is missing ImagePositionPatient tag, cannot be sorted",
    "ImageTooLarge": "Slice in series has height or width greater than 512",
    "ImageTooSmall": "Slice in series has height or width less than 512 and needs to be padded before inference",
    "DicomDecompressionError": "Dicom Decompression Error: Series contained pixel values >12000 or <-11000",
    "WrongOrientation": "Series is non-axial orientation",
}


class CheckSeriesLength(Component):
    def __init__(self, min_series_length: int = 20):
        """

        Filter object that ensures each series in a study is longer than a certain threshold.

        Parameters:
        -----------
        min_series_length: int
            Minimum length used to disqualify series from further analysis.

        """
        super().__init__()
        self.min_series_length = min_series_length

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        files = series_summary["files"]

        if len(files) >= self.min_series_length:
            return series_summary
        else:
            series_summary["valid"] = False
            series_summary["reason"] = _ERROR_MESSAGES["seriesLength"]

        return series_summary


class CheckModality(Component):
    def __init__(self):
        """

        Filter object that ensures each series in a study is only of modality type CT.

        """
        super().__init__()

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        datasets = series_summary["datasets"]

        dcm = datasets[0]

        try:
            modality = dcm.Modality
        except Exception as e:
            return self.report_exception(e, series_summary)

        if modality != "CT":
            series_summary["valid"] = False
            series_summary["reason"] = _ERROR_MESSAGES["modality"]

        return series_summary


class CheckCircleCrop(Component):
    def __init__(self, threshold: int = 50):
        """Filter object that ensures each series in a study hasn't been cropped.

        Parameters
        ----------
        threshold: int
            Difference between maximum and minimum series reconstruction
            diameters to determine cropping.

        """
        super().__init__()
        self.threshold = threshold

    def get_rec_diams(self, study_summary: StudySummary):
        self.rec_diams = defaultdict(float)

        for k, v in study_summary["UIDs"].items():

            if v["datasets"] and v["valid"]:
                dcm = v["datasets"][0]

                try:
                    rec_diam = dcm.ReconstructionDiameter
                    if rec_diam:
                        self.rec_diams[k] = rec_diam
                        study_summary["UIDs"][k]["recDiam"] = rec_diam
                except Exception as e:
                    study_summary["UIDs"][k] = self.report_exception(e, v)

        return study_summary

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        if (
            max(list(self.rec_diams.values())) - min(list(self.rec_diams.values()))
            > self.threshold
        ):
            if self.rec_diams[series_summary["series_uid"]] + self.threshold < max(
                list(self.rec_diams.values())
            ):
                series_summary["valid"] = False
                series_summary["reason"] = _ERROR_MESSAGES["circleCrop"]

        return series_summary

    def apply(self, study_summary: StudySummary):

        study_summary = self.get_rec_diams(study_summary)
        for series_uid, series_summary in study_summary["UIDs"].items():
            self.rec_diams[series_uid]
            study_summary["UIDs"][series_uid] = self.F(series_summary)

        return study_summary


class CheckEntangledSeries(Component):
    def __init__(self):
        """

        Filter object that checks if any series in a study consist of multiple
        seperate series entangled together.

        """
        super().__init__()

    def check_sorted(self, l):
        """

        Helper function to check if a python list is sorted

        """
        l = [float(e) for e in l]
        if l == sorted(l):
            return True
        if l == sorted(l, reverse=True):
            return True
        return False

    def check_out_of_plane_dimension(self, dcms, threshold=20):
        """

        Helper function to determine which index of the ImagePatientPatient
        varies most quickly between slices.

        """
        pos0 = [dcm.ImagePositionPatient[0] for dcm in dcms]
        pos1 = [dcm.ImagePositionPatient[1] for dcm in dcms]
        pos2 = [dcm.ImagePositionPatient[2] for dcm in dcms]

        flag = -1
        if len(np.unique(pos0)) >= threshold:
            flag = 0
        elif len(np.unique(pos1)) >= threshold:
            flag = 1
        else:
            flag = 2
        return flag

    def check_monotonic(
        self,
        datasets,
        attribute,
        secondary_attribute="ImagePositionPatient",
        reverse_t=False,
    ):
        """

        Helper function to check if two DICOM tags share a monotonic
        relationship for all slices in in a given series. Function checks for
        both increasing and decreasing monotonic relationship. If decreasing
        monotonic relationship, the series needs to be reversed when sorted.

        """
        self.d = self.check_out_of_plane_dimension(datasets)

        datasets1 = sorted(
            datasets,
            key=lambda dcm: (
                float(getattr(dcm, attribute)),
                float(getattr(dcm, secondary_attribute)[self.d]),
            ),
            reverse=reverse_t,
        )

        pos1 = [dcm.ImagePositionPatient[self.d] for dcm in datasets1]

        if self.check_sorted(pos1):
            return True, False

        datasets2 = sorted(
            datasets,
            key=lambda dcm: (
                float(getattr(dcm, attribute)),
                -float(getattr(dcm, secondary_attribute)[self.d]),
            ),
            reverse=reverse_t,
        )

        pos2 = [dcm.ImagePositionPatient[self.d] for dcm in datasets2]

        if self.check_sorted(pos2):
            return True, True

        return False, False

    def disentangle_multiple_series(self, dcms, series_files_list):
        """Helper function to identify the subsets of series entangled together.

        These are identified by breakpoints in the ImagePositionPatient value,
        which should always be increasing when sorted by ContentTime. If
        self.reverse is True, this means that the patient was scanned
        toe-to-head instead of head-to-toe, so the series needs to be sorted in
        reverse order before the breakpoints and subseries are identified.

        """
        if self.reverse:
            series_files_list = [
                s
                for _, s in sorted(
                    zip(dcms, series_files_list),
                    key=lambda pair: (
                        float(pair[0].ContentTime),
                        -float(pair[0].ImagePositionPatient[self.d]),
                    ),
                )
            ]
            dcms = sorted(
                dcms,
                key=lambda dcm: (
                    float(dcm.ContentTime),
                    -float(dcm.ImagePositionPatient[self.d]),
                ),
            )

        else:
            series_files_list = [
                s
                for _, s in sorted(
                    zip(dcms, series_files_list),
                    key=lambda pair: (
                        float(pair[0].ContentTime),
                        float(pair[0].ImagePositionPatient[self.d]),
                    ),
                )
            ]
            dcms = sorted(
                dcms,
                key=lambda dcm: (
                    float(dcm.ContentTime),
                    float(dcm.ImagePositionPatient[self.d]),
                ),
            )

        pos = [dcm.ImagePositionPatient[self.d] for dcm in dcms]

        # find the breakpoints
        increasing = True
        if pos[0] > pos[1]:
            increasing = False

        prev = pos[0]
        breakpoints = []
        for idx, p in enumerate(pos[1:]):
            if increasing and p < prev:
                breakpoints.append(idx + 1)
            elif not increasing and p > prev:
                breakpoints.append(idx + 1)
            prev = p

        # break the sub-series
        subseries = []
        subdatasets = []
        prev_b = 0
        for b in breakpoints:
            subseries.append(series_files_list[prev_b:b])
            subdatasets.append(dcms[prev_b:b])
            prev_b = b
        subseries.append(series_files_list[prev_b:])
        return subseries, subdatasets

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        files = series_summary["files"]
        series_dataset = series_summary["datasets"]

        try:
            self.content_time_monotonic, self.reverse = self.check_monotonic(
                series_dataset, attribute="ContentTime"
            )
            self.acquisition_time_monotonic, self.reverse = self.check_monotonic(
                series_dataset, attribute="AcquisitionTime"
            )
        except Exception as e:
            return self.report_exception(e, series_summary)
        if not self.content_time_monotonic and not self.acquisition_time_monotonic:
            try:
                subseries, subdatasets = self.disentangle_multiple_series(
                    series_dataset, files
                )
            except Exception as e:
                series_summary = self.report_exception(e, series_summary)
                series_summary["reason"] += _ERROR_MESSAGES["entangledSeries"]
                return series_summary

            series_summary["files"] = subseries
            series_summary["datasets"] = subdatasets
            series_summary["valid"] = False
            series_summary["reason"] = _ERROR_MESSAGES["entangledSeries"]

        return series_summary

    def apply(self, study_summary: StudySummary):

        disentangled_series = StudySummary()

        for series_uid, series_summary in study_summary["UIDs"].items():

            study_summary["UIDs"][series_uid] = self.F(series_summary)

            if (
                study_summary["UIDs"][series_uid]["reason"]
                == _ERROR_MESSAGES["entangledSeries"]
            ):

                for idx, (s, d) in enumerate(
                    zip(
                        study_summary["UIDs"][series_uid]["files"],
                        study_summary["UIDs"][series_uid]["datasets"],
                    )
                ):
                    new_series_summary = SeriesSummary()
                    new_series_summary.set_files(s)
                    new_series_summary["datasets"] = d
                    new_series_summary["PrimaryUID"] = series_uid + "_sub_" + str(idx)
                    disentangled_series["UIDs"][
                        series_uid + "_sub_" + str(idx)
                    ] = new_series_summary
                    disentangled_series["UIDs"][series_uid + "_sub_" + str(idx)][
                        "droppable_keys"
                    ].append("datasets")
                study_summary["UIDs"][series_uid].set_files([])
                study_summary["UIDs"][series_uid]["datasets"] = []

        study_summary["UIDs"].update(disentangled_series["UIDs"])

        return study_summary


class CheckType(Component):
    def __init__(self):
        """

        Filter object that checks that no series contain images of type DERIVED or GSI MD.

        """
        super().__init__()

    def check_modified_att(self, study_summary: StudySummary):
        """

        Helper function to gather additional information from metadata that is
        helpful in identifying why certain scans are marked as DERIVED.

        """
        for k, v in study_summary["UIDs"].items():
            series_modified_attribute_tags = []
            for dcm in v["datasets"]:
                try:
                    study_summary["UIDs"][k]["manufacturer"] = dcm.Manufacturer
                except Exception as e:
                    study_summary["UIDs"][k]["manufacturer"] = str(e)
                try:
                    seq = dcm.OriginalAttributesSequence[0].ModifiedAttributesSequence[
                        0
                    ]
                    seq_json = seq.to_json()
                    series_modified_attribute_tags.append(seq_json)
                except Exception as e:
                    study_summary["UIDs"][k]["modifiedAttributesSequence"] = str(e)
            study_summary["UIDs"][k]["modifiedAttributesSequence"] = dict(
                enumerate(np.unique(series_modified_attribute_tags))
            )
        return study_summary

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        datasets = series_summary["datasets"]

        dcm = datasets[0]

        try:
            image_type = dcm.ImageType
        except Exception as e:
            return self.report_exception(e, series_summary)

        if "DERIVED" in image_type:
            series_summary["valid"] = False
            series_summary["reason"] = _ERROR_MESSAGES["imageType"]["DERIVED"]
            try:
                derivation_description = dcm.DerivationDescription
                if (
                    "lossless" in derivation_description
                    and "lossy" not in derivation_description
                ):
                    series_summary["reason"] = (
                        "series marked as derived but lossless compression was the "
                        "only identified derivation, may still be ok to use, "
                        "requires manual review"
                    )
                    series_summary["derivationDescription"] = derivation_description
                else:
                    series_summary["derivationDescription"] = derivation_description
            except Exception as e:
                series_summary["derivationDescription"] = str(e)

        if "GSI MD" in image_type:
            series_summary["valid"] = False
            series_summary["reason"] = _ERROR_MESSAGES["imageType"]["GSI MD"]

        return series_summary

    def apply(self, study_summary: StudySummary):

        self.check_modified_att(study_summary)

        for series_uid, series_summary in study_summary["UIDs"].items():
            study_summary["UIDs"][series_uid] = self.F(series_summary)

        return study_summary


class CheckPixelArraySize(Component):
    def __init__(self):
        """

        Filter object that checks if slices in series have non-axial dimensions
        other than standard 512x512. If either dimension is >512, series is
        marked as invalid. If either dimension is <512, series is marked as
        requiring preprocessing.

        """
        super().__init__()

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        series_dataset = series_summary["datasets"]

        for dcm in series_dataset:

            if dcm.Rows > 512 or dcm.Columns > 512:
                series_summary["valid"] = False
                series_summary["reason"] = _ERROR_MESSAGES["ImageTooLarge"]

            elif dcm.Rows < 512 or dcm.Columns < 512:
                series_summary["valid"] = True
                series_summary["preProcessingRequired"] = True
                series_summary["reason"] = _ERROR_MESSAGES["ImageTooSmall"]

        return series_summary


class CheckDICOMDecompressionError(Component):
    def __init__(self):
        """

        Filter object that checks if any DICOM files in a series had problems
        when being decompressed. Slices with problems are defined as those
        having pixel values <-11000 or >12000.

        """
        super().__init__()

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        series_dataset = series_summary["datasets"]

        for dcm in series_dataset:

            try:
                data = dcm.pixel_array
            except Exception as e:
                return self.report_exception(e, series_summary)

            if np.min(data) < -11000 or np.max(data) > 12000:
                series_summary["valid"] = False
                series_summary["reason"] = _ERROR_MESSAGES["DicomDecompressionError"]

        return series_summary


class CheckOrientation(Component):
    def __init__(self):
        """Check orientation of the images.

        Checks that images are axial and supine (within a small tolerance).

        """
        self.optimal_patient_orientation = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        super().__init__()

    def check_threshold(self, image_orientation_patient):
        dot_prod = np.dot(
            np.absolute(image_orientation_patient), self.optimal_patient_orientation
        )
        if abs(2 - dot_prod) <= 0.05:
            return True
        return False

    def F(self, series_summary: SeriesSummary):

        if not self.check_valid(series_summary):
            return series_summary

        datasets = series_summary["datasets"]

        dcm = datasets[0]

        try:
            image_orientation_patient = dcm.ImageOrientationPatient
        except Exception as e:
            return self.report_exception(e, series_summary)

        if not self.check_threshold(image_orientation_patient):
            series_summary["valid"] = False
            series_summary["reason"] = _ERROR_MESSAGES["WrongOrientation"]

        return series_summary

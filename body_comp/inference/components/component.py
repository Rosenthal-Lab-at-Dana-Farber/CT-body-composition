from abc import ABC, abstractmethod

from collections import defaultdict
from functools import reduce
import operator
from typing import Union


class StudySummary(defaultdict):

    def __init__(self, files=None, series=True):
        super().__init__(list)
        if files:
            self["files"] = files
        if series:
            self["UIDs"] = defaultdict(SeriesSummary)
        else:
            self["UIDs"] = defaultdict(SliceSummary)


class SeriesSummary(defaultdict):

    def __init__(self):
        super().__init__(list)
        self["valid"] = True
        self["reason"] = "no error"

    def set_files(self, f):
        self["files"] = f


class SliceSummary(defaultdict):
    """
    Enhanced dictionary object for handling analysis of single CT slices
    """

    def __init__(self):
        super().__init__(list)
        self["valid"] = True
        self["reason"] = "no error"


class Component(ABC):
    """Base class for all components.

    Each Component must operate on a DICOM image (or NIFTI?), maybe dict
    instead?

    """

    def __init__(self):
        self.last = False

    def __repr__(self):
        return "Base class for all Components"

    def check_valid(self, series_summary: SeriesSummary):
        """Check if a series has already been marked by a previous filter.

        If yes, that series is skipped for further processing.

        Parameters
        ----------
        series_summary: SeriesSummary
            SeriesSummary object containing data about that series aggregated
            by previous Components in the pipeline.

        Returns
        -------
        bool

        """
        if series_summary["valid"]:
            return True
        return False

    # mark last component in a pipeline so it drops unnecessary data from
    # memory before saving cohort summary to JSON file
    def set_last(self):
        self.last = True

    def drop_keys(self, study_summary: StudySummary):

        for _, series_summary in study_summary["UIDs"].items():
            # remove duplicate keys from the series summary's droppable_keys field
            series_summary["droppable_keys"] = list(
                set(series_summary["droppable_keys"])
            )

            for key in series_summary["droppable_keys"]:
                try:
                    keys = key.split("/")
                    if len(keys) == 1:
                        series_summary.pop(keys[0])
                    else:
                        reduce(operator.getitem, keys[:-1], series_summary).pop(
                            keys[-1]
                        )
                except:
                    print(
                        f"drop_keys was called but series_summary for series "
                        f"{series_summary['PrimaryUID']} has no {key} key"
                    )

        return study_summary

    def report_exception(self, e, series_summary: SeriesSummary):
        series_summary["valid"] = False
        series_summary["reason"] = str(e)
        return series_summary

    @abstractmethod
    def F(self, summary: Union[StudySummary, SeriesSummary]):
        """functional implementation"""
        pass

    def apply(self, study_summary: StudySummary):
        """Apply Component on study and return results in updated study summary.

        Default functionality is to iterate over each series in the study
        summary and update its corresponding series summary.

        For certain components, additional functionalities may be required.

        """
        for series_uid, series_summary in study_summary["UIDs"].items():
            study_summary["UIDs"][series_uid] = self.F(series_summary)
        return study_summary

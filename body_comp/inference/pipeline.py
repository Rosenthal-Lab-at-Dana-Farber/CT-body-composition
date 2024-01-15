import json
import os

from collections import defaultdict

from .components.component import Component, StudySummary
from .utils import default


class Pipeline:
    """Object to compose a sequence of components

    Args:
        component_sequence (list): sequence of components to be consecutively applied.
            List of `Component` objects
    """

    def __init__(self, component_sequence=None):
        assert component_sequence is None or all(
            [isinstance(t, Component) for t in component_sequence]
        ), "All elements in input list must be of type Component"
        self.components = component_sequence
        if self.components is not None:
            self.components[-1].set_last()
        self.cohort_summary = defaultdict(StudySummary)

    def __len__(self):
        if self.components is not None:
            return len(self.components)
        return 0

    def __repr__(self):
        if self.components is None:
            return "Pipeline()"
        else:
            out = "Pipeline([\n"
            for t in self.components:
                out += f"\t{repr(t)},\n"
            out += "])"
            return out

    def apply(self, root, series=True):
        """Apply component sequence to directory recursively starting at root.

        This function assumes any sets of files found in different subfolders
        belong to different studies.

        Parameters
        ----------
        root: str
            Path to root folder of DICOM cohort.

        """
        if not self.components:
            raise RuntimeError(
                "No components specified. Pipeline object must be initialized "
                f"with components that will be applied to {root}."
            )
        else:
            count = 0
            for _, (root, _, files) in enumerate(os.walk(root, followlinks=True)):
                if files:
                    paths = [os.path.join(root, file) for file in files]
                    count += 1
                    print(f"Analyzing study {count}: {root}")
                    study_summary = StudySummary(files=paths, series=series)
                    for t in self.components:
                        study_summary = t.apply(study_summary)
                        if t.last:
                            study_summary = t.drop_keys(study_summary)
                    self.cohort_summary[root] = study_summary

    def get_cohort_summary(self):
        return self.cohort_summary

    def save_cohort_summary(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.cohort_summary, outfile, default=default)

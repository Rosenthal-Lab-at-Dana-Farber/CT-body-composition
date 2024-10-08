import json
import os
from pathlib import Path

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

    def apply(
        self,
        root,
        series=True,
        study_depth=None,
        study_list=None,
        summary_path=None,
        keep_existing=False,
    ):
        """Apply component sequence to directory recursively starting at root.

        This function assumes any sets of files found in different subfolders
        belong to different studies.

        Parameters
        ----------
        root: str
            Path to root folder of DICOM cohort.
        series: bool
            Whether to apply the pipeline to each series as a whole (if True)
            or each individual file/slice (if False).
        study_depth: int | None
            If None, every subdirectory at any level of the hierarchy under
            'root' that contains files is considered a study. If study_depth is
            a non-negative integer, then each directory that number of levels
            below route is considered a study. E.g. if study_depth is 0, the
            root directory itself is a single study. If study_depth is 1, each
            sub-directory of root is considered a study. If study_depth is 2,
            each sub-directory of a sub-directory of root is considered a
            study, etc. If this option is used, any file at any level under a
            study directory is included (for example, files may be grouped into
            series directories under the study level). Incompatible with
            study_list.
        study_list: str | None
            Path to a file containing a list of studies to process. This
            allows you to specify that only a subset of studies under the
            root be processed (by default all are processed). The file
            should be a plain text. Each line within the file should contain
            the path to a single study to be processed, given relative to
            the root directory. Incompatible with study_depth.
        summary_path: str | None
            If non-None, the summary JSON is written out to this path after
            each study is processed.
        keep_existing: bool
            If True and an existing summary file exists, reads the existing
            summary file, skips any studies already in it, and appends new
            results to the existing file. If False, or no summary file exists,
            processes all studies in the root directory.

        """
        if not self.components:
            raise RuntimeError(
                "No components specified. Pipeline object must be initialized "
                f"with components that will be applied to {root}."
            )

        if not Path(root).exists():
            raise FileNotFoundError(
                f"The specified root directory does not exist {str(root)}"
            )

        if study_list is not None and study_depth is not None:
            raise TypeError(
                "'study_list' and 'study_depth' are incompatible."
            )

        if study_depth is not None:
            if study_depth == 0:
                study_dirs = [Path(root)]
            elif study_depth > 0:
                pattern = "/".join(["*"] * study_depth)
                study_dirs = [
                    p for p in Path(root).glob(pattern) if p.is_dir()
                ]
            else:
                raise ValueError("Study depth must be a non-negative integer")
        elif study_list is not None:
            with open(study_list, 'r') as f:
                study_dirs = [
                    Path(root).joinpath(p.strip()) for p in f.readlines()
                ]

            for d in study_dirs:
                if not d.exists():
                    raise FileNotFoundError(
                        f"Study directory {str(d)} not found."
                    )
        else:
            study_dirs = [
                Path(study_dir)
                for study_dir, _, files in os.walk(root, followlinks=True)
                if files
            ]

        print(f"Found {len(study_dirs)} studies in root directory.")

        if keep_existing and os.path.exists(summary_path):
            # Read existing summary file and remove cases that have already
            # been processed from the to-do list
            self.read_cohort_summary(summary_path)
            print(
                f"Loaded existing results from {len(self.cohort_summary)} "
                "studies"
            )
            study_dirs = [
                d for d in study_dirs if str(d) not in self.cohort_summary
            ]

        for count, study_dir in enumerate(study_dirs):
            print(f"Analyzing study {count}/{len(study_dirs)}: {study_dir}")

            if study_depth is None and study_list is None:
                # Only files directly in the directory
                paths = [str(p) for p in study_dir.iterdir() if p.is_file()]
            else:
                # Any file anywhere under the directory
                paths = []
                for dirpath, _, files in os.walk(study_dir):
                    for f in files:
                        paths.append(os.path.join(dirpath, f))

            study_summary = StudySummary(files=paths, series=series)
            for t in self.components:
                study_summary = t.apply(study_summary)
                if t.last:
                    study_summary = t.drop_keys(study_summary)
            self.cohort_summary[str(study_dir)] = study_summary

            # Write out the summary on every loop
            if summary_path is not None:
                self.save_cohort_summary(summary_path)

    def get_cohort_summary(self):
        return self.cohort_summary

    def save_cohort_summary(self, filename):
        with open(filename, "w") as outfile:
            json.dump(
                self.cohort_summary,
                outfile,
                default=default,
                indent=2
            )

    def read_cohort_summary(self, filename):
        with open(filename, "r") as infile:
            self.cohort_summary = json.load(infile)


# Running Inference

### Input Data

 The input directory for any inference run should be the root folder of the
cohort. A typical directory structure might look like the following:


                      cohort_root_folder
                               |
                _________________________________
                |              |                |
              study_1       study_2        study_3
                |               |               |
             1.dcm            1.dcm           1.dcm
             2.dcm            2.dcm           2.dcm
             3.dcm            3.dcm           3.dcm


Each study in the cohort will contain a batch of
[DICOM](https://www.dicomstandard.org/about) files, which will make up one or
more series.

### Running the Process (Command Line Interface)

The primary script for running inference, `analyze_cohort.py``, runs on cohorts
consisting of DICOM files.  Most users will find that this script is suitable
for their needs. If you need advanced configuration, consider using the Python
API directly (see below).

Basic usage looks like this:

```bash
$ python analyze_cohort.py /path/to/root/ /path/to/output/folder
```

See notes below for more information on the config file to set up the
configuration file before running.

At the end of the process, the results directory will contain several artifacts:

`previews/` - This directory contains preview images that may be used for
efficient visual checking of the results. There is one preview png file per
*series* that was successfully processed by the algorithm (there are often
multiple such series per study). Each file is named
`{MRN}_{ACC}_{SeriesInstanceUID}_preview.png` (or
`{StudyName}_{SeriesInstanceUID}_preview.png`).

`ready_for_qa/` - This directory consists of a subset of the images in the
`previews` folder after a filtering process has been applied. The filtering
process removes likely slice selection failures, and selects the most
appropriate series per study if there are multiple series that successfully ran
for a particular study.

`results.json` - This file contains the full results from the analysis in JSON format. 
Each key in the file is a study, which contains study-level information pulled out by the Components as well as nested dictionaries containing Series/Slice level information and analysis results.

`qc_results.csv` - This file contains QC results about the cohort. Each row in the file is a series identified within the cohort, and the result of whether the series passed/failed. as well as it's potential reason for failing, is given.

`qc_summary.csv` - This file summarizes the QC results and gives the frequency of each reason why certain series were rejected.

`results.csv` - This file lists a summary of the results on
every *series* successfully processed by the model, including the area of the
body composition compartments and their Hounsfield unit statistics. It is a
subset of the information in the JSON files.

`filtered_results.csv` - This file consists of a subset of the rows of the
`results.csv` file after a filtering process has been applied. The filtering
process removes likely slice selection failures, and selects the most
appropriate series per study if there are multiple series that successfully ran
for a particular study.

### Config File

The process requires a configuration file in the JSON format to set up certain
parameters of the inference process. The default configuration file is provided
in the configs subdirectory of the package. You will need to edit this to
specify the locations of your trained models.

Here is the layout of an example configuration file:

```json
{
    "sigmoid_output": true,
    "slice_selection_weights": "/path/to/some/model.hdf5",
    "slice_params": {
        "L3": {
            "slice_selection_model_output_index": 0,
            "class_names": [
                "muscle",
                "subcutaneous_fat",
                "visceral_fat"
                ],
            "model_weights": "/path/to/some/other_model.hdf5",
            "regression_plot_colour": "red"
        }
    }
}
```

Explanations of each of these parameters are found below:

```
slice_selection_weights: string
    Path to hdf file containing the model weights for the slice selection model.
slice_params: dict
    Python dictionary containing a mapping from string (slice name) to dictionary.
    The lower dictionary has the following entries:
        'slice_selection_model_output_index': int
            Which output of the slice selection model relates to this slice (only relevant if sigmoid_output is
            True)
        'z_pos': float
            The position of this slice in the latent space (only relevant if sigmoid_output is False)
        'class_names': list of strings
            Names of the classes present in the segmentation model of this slice, excluding the background
            class. Should match the number of channels of the segmentation model (-1 due to background class).
            List should be in increasing order of channel index.
        'model_weights': string
            Path to hdf file containing the model weights for the segmentation model to use for this slice.
        'regression_plot_colour': string
            A colour (as recognised by a matplotlib plot command) to use for this slice on the output regression
            plots
    The default value of ``slice_params`` is configured to run on an L3 slice to segment three compartments:
    muscle, visceral fat, and subcutaneous fat.
sigmoid_output: bool
    Set to true if the slice selection model outputs a true sigmoid range (between 0 and 1 rather than -1
    and 1) for each target slice. If false, the slice selection model outputs a single number in a 1D space for
    each input slice, which is compared to the 'z_pos' field of 'slice_params' to perform slice selection.
```


### Additional Options

There are a number of further options to customize the behavior of this process.
They may be passed as command-line arguments to the `analyze_cohorts.py` file:

`--num_threads`, `-t` - The number of parallel threads to use to read in DICOM
files (typically the most time intensive step of the processing especially if
the files are being read from a remote file system). You should choose this
appropriately based on your hardware. Using a higher number if multiple CPU
cores are available will usually speed up processing significantly.

`--segmentation_range`, `-r` - The range either side of the selected slice (in
mm) to perform segmentation on for multislice analysis. If this is specified,
then the slices are selected as usual, but then any slice that lies within the
given distance of the selected slice is segmented, and the results are
averaged. This usually gives a more robust result, but setting it too high will
cause the model to segment areas that it wasn't trained to segment. If this
option is chosen, there is an additional sub-directory of the output directory
called `all_slices`, which stores segmentation masks and original images for
every output slice in `.png` format in sub-directories named by
`{MRN}_{ACC}_{SeriesInstanceUID}`. The filename of the `.png` images matches
that the position in the JSON file. The preview image contains just the chosen
center slice.  Note that this option requires that slice selection is *not*
disabled with the --no_slice_selection option.

`--keep_existing`, `-k` - This flag is used to carry on a process that was
previously interrupted. If the same output directory as a previous run is used,
the process will not re-process studies that already have results.

`--dicom_seg`, `-d` - If specified, the segmentations will also be output in
DICOM segmentation format. This is a standard format that can be read and
displayed by some DICOM viewers.

`--min_slices_per_series`, `-m` - Reject series with fewer than this number of
images. Useful for rejecting small localizer series. Default: 20

`--slice_selection`, `-s` - Do not perform slice selection as part of analysis
pipeline and instead run the segmentation on every slice. Note that model
accuracy deteriorates away from the slices it was trained on (L3 for the
default model). By default, slice selection is performed and the segmentation
is performed only on the selected slices.

`--study_depth`, `-e` - Depth of study directories under root.  If not
provided, every subdirectory at any level of the hierarchy under 'root' that
contains files is considered a "study". If `study_depth` is a non-negative
integer, then each directory that number of levels below `root` is considered a
study.  E.g. if `study_depth` is 0, the root directory itself is a single
study.  If `study_depth` is 1, each sub-directory of `root` is considered a
study. If `study_depth` is 2, each sub-directory of a sub-directory of root is
considered a study, etc. If `study_depth` is used, any file at any level under
a study directory is included (for example, files may be grouped into series
directories under the study level and still be processed as a single study).
Incompatible with `study_depth`.

`--study_list`, `-l` - Path to a file containing a list of studies to process.
This allows you to specify that only a subset of studies under the root be
processed (by default all are processed). The file should be a plain text. Each
line within the file should contain the path to a single study to be processed,
given relative to the root directory. Incompatible with `study_depth`.

### Python API

If you want to customize the operation of the pipeline, particularly the
filtering and QA components, you can write your own Python code that makes use
of our Python API within the `body_comp` module.

#### Pipeline Structure

The primary building block of our inference framework is a **Component** object. A Component represents a mechanism that performs some type of action on a study. The three types of actions currently supported are:

1. Preprocessing
2. Filtering
3. Analysis

Components run analysis on one study at a time, identifying all of the series in that study through their [SeriesUID](https://dicom.innolitics.com/ciods/cr-image/general-series/0020000e) attribute. Each study is represented as a StudySummary, a customized dictionary that each Component receives and edits based on its specific functionality.

All components defined according to our API can be chained together similar to torch [transforms](https://pytorch.org/vision/stable/transforms.html). This way, our inference framework is fully modularized, and any Component can be added or removed based on the needs of the user without affecting other aspects of the pipeline.

#### DICOM QC

 Before we can run inference, we need to run quality control (QC) on the data in the cohort. In order to do this, we have written a special type of component called a filter that identifies common problems we've seen in newly received DICOM cohorts. In additiona to custom user-specified functionality, every filter marks each series as passing or failing based on the specific parameters of the filter.

Filters can be added/ommitted based on specific properties that we care about for a given cohort.

#### Creating a Pipeline

In order to chain multiple Component objects together, we use a Pipeline object, which ensures all Components adhere to our API. This allows users to write complex analysis pipelines using a few lines of code. The first step is to define the pipeline's Component sequence:

```python
transforms = [
    FindSeries(),
    FillMetadata(),
    CheckType(),
    CheckModality(),
    CheckSeriesLength(),
    WindowAndShiftDicomSeries(),
    SliceSelector(
        **slice_selector_config,
        num_threads=num_threads,
    ),
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
```

Once we've determined the Component sequence we want, we instantiate a Pipeline object.

```python
pipeline = Pipeline(transforms)
```

Once this is done, we apply the pipeline to a root folder.

```python
pipeline.apply("/path/to/root")
```

As can be seen above, this reduces the arduous process of manually QCing an entire cohort down to a few lines of code.

One property to note is that a pipeline object looks through directories recursively, and assumes every folder it identifies with files in it is a unique study.

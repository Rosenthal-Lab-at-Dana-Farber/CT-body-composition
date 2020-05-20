# Running Inference

Inference is the process of running the existing trained model on new data.

### Input Data

The inference process works directly on the data in DICOM format. However, the
DICOM files must be organized into studies in a particular format. The DICOM
files for each study should be placed into a single directory, where the name
of the directory is the MRN of the study followed by the accession number of
the study, separated by an underscore.  For example `0123467_135792468`. Within
the study directory, the names of the DICOM files do not matter.

If you do not have the data placed in this format, there is a script in the
`bin` directory that will created an organized copy of your data for you.

```
$ python organize_inference_data.py /path/to/existing_data_directory /path/to/new/organized/directory
```

In addition, there should be a CSV file with columns named `MRN` and `ACC`,
which list the MRN and accession numbers that you wish to process. E.g.,

```
MRN,ACC
012345,678910
54321,109876
314159,26535
```

### Running the Process

The `run_from_csv.py` script is the main script for performing inference. It
takes in a CSV file of MRNs and ACCs (described above), and data directories,
runs inference on every study, including the series selection, slice selection,
and segmentation steps, and then outputs the results into a results directory.

Basic usage looks like this:

```
python run_from_csv.py my_csv_file.csv /path/to/results/directory /path/to/input/directory1 /path/to/input/directory2
```

Note that there can be an arbitrary number (one or more) of input directories,
each with the studies laid out in the `MRN_ACC` format. This allows for
processing data from multiple data pulls at once.

At the end of the process, the results directory will contain several artifacts:

`json_files/` - This directory contains the full results from the body
composition analysis in JSON format.  There is one file per study, named with
the study `MRN_ACC.json`.

`previews/` - This directory contains preview images that may be used to check
the results visually efficiently. There is one preview png file per *series*
that was successfully processed by the algorithm (there are often multiple such
series per study). Each file is named `{MRN}_{ACC}_{SeriesInstanceUID}.png`.

`run_log.csv` - This file contains the basic results from running the
model.  For each study listed in the input csv file, it lists whether the
relevant DICOM data was found successfully, how many series from this study (if
any) the model was able to run on successfully.

`summary.csv` - This file lists a summary of the results of the results on
every *series* successfully processed by the model, including the area of the
body composition compartments and their Hounsfield unit statistics. It is a
subset of the information in the JSON files.

`filtered_summary.csv` - This file consists of a subset of the rows of the
`summary.csv` file after a filtering process has been applied. The filtering
process removes likely slice selection failures, and selects the most
appropriate series per study if there are multiple series that successfully ran
for a particular study.

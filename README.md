# CT Body Composition

This repository provides code for training and running body composition estimation models on abdominal CT scans.

### Installation

**Important** this repository uses git-lfs (large file storage) to store model weights efficiently in git.
Before cloning this repository, make sure that you have git-lfs (large file system) installed.
See [here](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) for help.

After cloning the repository, you have two options for setting up the environment.
You may install all the necessary components directly on your system, or if you have docker on your machine,
you may build a docker image that contains all the necessary requirements.

##### Direct Installation

Part of this repository contains a Python package called `body_comp`, which should be installed before running any
of the code.
Once you have cloned this repository you can install the `body_comp` package and all its dependencies by running
this command from the root of the repository:

```
$ pip install .
```

Note that this will not install the `gdcm` python package, because it is not in the PyPI package repository.
Having this package installed is important to be able to decompress some of the less common DICOM transfer
syntaxes (compression methods). If you are using Anaconda, you can install the `gdcm` package from their repositories.
On a Mac, you can install `gdcm` with:

```
brew install gdcm --with-python3
```

Alternatively, you can build the `gdcm` package from source. See the Dockerfile for an example of this process.

##### Building the Docker Image



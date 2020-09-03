# CT Body Composition

This repository provides code for training and running body composition estimation models on abdominal CT scans. 
It is a branch derived from Chris Bridge's code at CCDS but is relocated here to allow for access and development.
We may have updates that we make pulled back into the main CCDS repository, but that is TBD.

All code and documentation from the primary repository are credited to Chris Bridge.


### Getting Started

**Important** this repository uses git-lfs (large file storage) to store model weights efficiently in git.
Before cloning this repository, make sure that you have git-lfs (large file system) installed.
See [here](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) for help.

After cloning the repository, you have two options for setting up the environment.
You may install all the necessary components directly on your system, or if you have docker on your machine,
you may build a docker image that contains all the necessary requirements.

See the documentation pages for further details:
* [Installation](docs/installation.md) - For installing directly on your system
* [Docker](docs/docker.md) - For building and using the docker image
* [Training](docs/training.md) - For training new models
* [Inference](docs/inference.md) - For running the model on new data

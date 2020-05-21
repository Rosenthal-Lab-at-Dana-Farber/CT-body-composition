# CT Body Composition

This repository provides code for training and running body composition estimation models on abdominal CT scans.

### Installation

**Important** this repository uses git-lfs (large file storage) to store model weights efficiently in git.
Before cloning this repository, make sure that you have git-lfs (large file system) installed.
See [here](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) for help.

After cloning the repository, you have two options for setting up the environment.
You may install all the necessary components directly on your system, or if you have docker on your machine,
you may build a docker image that contains all the necessary requirements.

See the documentation pages for further details:
* [Installation](docs/installation.md) - For installing directly on your system
* [Docker](docs/docker.md) - For building and using the docker image
* [Inference](docs/inference.md) - For running the model on new data

# Start with the tensorflow base image
FROM tensorflow/tensorflow:1.15.0-gpu-py3
MAINTAINER Christopher Bridge

WORKDIR /

# If running on a Debian-based machine with a GPU uncomment the below 2 lines to ensure that the GPG keys are in line with NVIDIA's newest rotation: https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904 ####

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

# Install the body composition code
COPY bin /body_comp/bin
COPY body_comp /body_comp/body_comp
COPY setup.py /body_comp/

RUN pip install -e /body_comp/

WORKDIR /body_comp/bin

# Start with the tensorflow base image
FROM tensorflow/tensorflow:2.15.0-gpu
MAINTAINER Christopher Bridge

WORKDIR /

# Install the body composition code
COPY bin /body_comp/bin
COPY body_comp /body_comp/body_comp
COPY setup.py /body_comp/

RUN pip install -e /body_comp/

WORKDIR /body_comp/bin

# Start with the tensorflow base image
FROM tensorflow/tensorflow:2.15.0-gpu
MAINTAINER Christopher Bridge

WORKDIR /

# Install version-locked versions of dependencies
RUN pip install --upgrade pip
RUN pip install \
    matplotlib==3.8.4 \
    scikit-image==0.23.2 \
    scipy==1.13.0 \
    pydicom==2.4.4 \
    pandas==2.2.2 \
    highdicom==0.22.0 \
    python-gdcm==3.0.23.1

# Install the body composition code
COPY bin /body_comp/bin
COPY body_comp /body_comp/body_comp
COPY setup.py /body_comp/

RUN pip install -e /body_comp/

WORKDIR /body_comp/bin

#############################################################################
# This Dockerfile is for use with Azure ML Darknet experiments.  It is meant
# to be the base of an Environment image for training object detection models.
# Darknet is a framework written in C/CUDA that must be built from source and
# here, the NVIDIA CUDA compiler is used allowing for accelerated training and 
# scoring.
# The base is from a build of darknet from https://github.com/AlexeyAB/darknet.
# Additionally, the following is built into the image:
# - Miniconda (custom Dockerfile method with Azure ML expects conda)
# - Azure ML SDK
#############################################################################
ARG BASE_IMAGE=nvidia/cuda:10.0-cudnn7-devel
FROM $BASE_IMAGE AS builder

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
      && apt-get install --no-install-recommends --no-install-suggests -y gnupg2 ca-certificates \
            git build-essential libopencv-dev python3-opencv \
      && rm -rf /var/lib/apt/lists/*

ARG SOURCE_BRANCH=master
ENV SOURCE_BRANCH $SOURCE_BRANCH

# 1/21/2021
ARG SOURCE_COMMIT="64efa72"
ENV SOURCE_COMMIT $SOURCE_COMMIT

RUN git clone https://github.com/AlexeyAB/darknet.git && cd darknet \
      && git checkout $SOURCE_BRANCH \
      && git reset --hard $SOURCE_COMMIT \
      && sed -i "s/GPU=0/GPU=1/g" Makefile \
      && sed -i "s/CUDNN=0/CUDNN=1/g" Makefile \
      && sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/g" Makefile \
      && sed -i "s/OPENCV=0/OPENCV=1/g" Makefile \
      && sed -i "s/AVX=0/AVX=1/g" Makefile \
      && sed -i "s/OPENMP=0/OPENMP=1/g" Makefile \
      && sed -i "s/LIBSO=0/LIBSO=1/g" Makefile \
      && sed -i "s/ARCH= -gencode arch=compute_30,code=sm_30 \\\//g" Makefile \
      && sed -i "s/      -gencode arch=compute_35,code=sm_35 \\\/ARCH= -gencode arch=compute_35,code=sm_35 \\\/g" Makefile \
      && make \
      && cp darknet /usr/local/bin \
      && cd .. && rm -rf darknet

FROM nvidia/cuda:10.0-cudnn7-runtime

RUN apt-get update \
      && apt-get install --no-install-recommends --no-install-suggests -y libopencv-highgui3.2 \
      wget bzip2 git \
      ffmpeg libsm6 libxext6 \
      && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/darknet /usr/local/bin/darknet

# Get Python in the form of Miniconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Update package installers
RUN conda update conda -y
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# Install Python packages
RUN pip install azureml-sdk==1.23.0
# RUN wget https://gist.githubusercontent.com/michhar/5eea9a65790debc9e53239b743ce167f/raw/ce7e0a2047213354d7a67af816e7dd0b1e60db7b/requirements-gpu-tflite.txt \
#     && pip install -r requirements-gpu-tflite.txt
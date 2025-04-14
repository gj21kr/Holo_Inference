# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.10-py3
FROM ${BASE_IMAGE} AS base

# Python Version
ARG PYTHON_VERSION=3.10
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV PYTHONPATH=""
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN python3 -m pip install --upgrade pip

# Holoscan dependencies (based on provided Dockerfile)
ARG ONNX_RUNTIME_VERSION=1.18.1
ARG GRPC_VERSION=1.54.2
ARG GXF_VERSION=4.1.1.4_20241210_dc72072

ENV CMAKE_PREFIX_PATH=""

# ONNX Runtime
RUN python3 -m pip install onnxruntime==${ONNX_RUNTIME_VERSION%%-*}

# gRPC
RUN python3 -m pip install grpcio==${GRPC_VERSION}

# GXF (requires manual download and install due to Holoscan's custom build)
ARG GXF_ARCH=arm64 # or arm64 depending on your architecture

RUN wget https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/gxf/gxf_${GXF_VERSION}_holoscan-sdk_${GXF_ARCH}.tar.gz \
    && tar -xzvf gxf_${GXF_VERSION}_holoscan-sdk_${GXF_ARCH}.tar.gz \
    && mv gxf_${GXF_VERSION}_holoscan-sdk_${GXF_ARCH} /opt/nvidia/gxf \
    && rm gxf_${GXF_VERSION}_holoscan-sdk_${GXF_ARCH}.tar.gz

ENV GXF=/opt/nvidia/gxf
ENV GXF_VERSION=${GXF_VERSION}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GXF}"
ENV PYTHONPATH="${PYTHONPATH}:/opt/nvidia/gxf/python"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install MONAI (ensure compatible PyTorch version)
RUN python3 -m pip install monai nibabel scikit-image SimpleITK pydicom pynrrd matplotlib joblib scipy

# Install Holoscan from holoscan:v2.5.0-igpu
RUN python3 -m pip install holoscan==2.5.0

# Verification - Optional but recommended
RUN python3 -c "import torch; print(torch.cuda.is_available()); import monai; import holoscan"

# cd /home/holoscan/Documents/Holo_Inference/
# python app.py -i /home/holoscan/Data/data/AVT/Dongyang/D1/D1.nrrd -o ./predicts/ -c avt
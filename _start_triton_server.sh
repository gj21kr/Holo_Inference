#!/bin/bash

# Triton 서버 시작 스크립트
MODEL_REPOSITORY=/triton_model_repository

# 도커를 통해 Triton 서버 실행
docker run --gpus all -it --rm \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $MODEL_REPOSITORY:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models \
    --log-verbose=1 \
    --strict-model-config=false \
    --exit-on-error=true
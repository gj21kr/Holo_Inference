#!/usr/bin/env python3

import os
import argparse
import logging
import importlib
import torch
import numpy as np
from pathlib import Path

from core.call import call_model
from utils.utils import load_saved_model

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("onnx_converter")

def load_config(config_name):
    try:
        _module = importlib.import_module(f"config.{config_name}")
        config = _module.config
        return config
    except ImportError as e:
        raise ImportError(f"설정 모듈을 가져올 수 없습니다: config.{config_name}") from e

def convert_torch_to_onnx(model, config, output_path, verbose=False):
    """PyTorch 모델을 ONNX 형식으로 변환합니다."""
    logger = logging.getLogger("onnx_converter")
    
    # 입력 크기 및 채널 가져오기
    input_shape = config.get("INPUT_SHAPE", [96, 96, 96])
    in_channels = config.get("CHANNEL_IN", 1)
    
    # 더미 입력 생성 (NCHW 형식)
    dummy_input = torch.randn(1, in_channels, *input_shape, device=next(model.parameters()).device)
    
    if verbose:
        logger.info(f"모델 변환 중... 입력 shape: {dummy_input.shape}")
    
    # ONNX 경로 확인
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 동적 배치 크기 설정
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
        
    # DataParallel 모델을 unwrap
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # ONNX로 내보내기
    torch.onnx.export(
        model,                     # 모델
        dummy_input,               # 모델 입력 (더미)
        output_path,               # 저장 위치
        export_params=True,        # 모델 파라미터 내보내기
        opset_version=13,          # ONNX 버전
        do_constant_folding=True,  # 상수 폴딩 최적화
        input_names=["input"],     # 입력 이름
        output_names=["output"],   # 출력 이름
        dynamic_axes=dynamic_axes, # 동적 축 설정
        verbose=verbose            # 자세한 정보 출력
    )
    
    logger.info(f"ONNX 모델이 저장되었습니다: {output_path}")
    return output_path

def main():
    logger = setup_logging()
    
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description='PyTorch 모델을 ONNX로 변환')
    parser.add_argument('-c', '--config', type=str, required=True, help='설정 파일 이름')
    parser.add_argument('-o', '--output', type=str, help='출력 ONNX 파일 경로')
    parser.add_argument('-v', '--verbose', action='store_true', help='자세한 출력')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 출력 경로 설정
    if args.output:
        output_path = Path(args.output)
    else:
        model_version = config.get("MODEL_VERSION", "default")
        output_dir = Path("triton_model_repository") / "monai_segmentation" / "1"
        output_dir.mkdir(parents=True, exist_ok=True)        
    
    # 모델 로드
    model = call_model(config)
    model = load_saved_model(config, model)
    model.eval()  # 평가 모드로 설정
    output_path = str(output_dir / f"{config['MODEL_VERSION']}.onnx")
    
    # ONNX로 변환
    convert_torch_to_onnx(model, config, output_path, args.verbose)
    
    logger.info("변환 완료!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import importlib
import torch
import numpy as np
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("triton_model_setup")

def load_config(config_name):
    """설정 파일을 로드합니다."""
    try:
        _module = importlib.import_module(f"config.{config_name}")
        config = _module.config
        return config
    except ImportError as e:
        raise ImportError(f"설정 모듈을 가져올 수 없습니다: config.{config_name}") from e

def load_model(config_name):
    """설정에 따라 모델을 로드합니다."""
    from core.call import call_model
    from utils.utils import load_saved_model
    
    config = load_config(config_name)
    model = call_model(config)
    model = load_saved_model(config, model)
    model.eval()  # 평가 모드로 설정
    return model, config

def convert_to_onnx(model, config, output_path, verbose=False):
    """PyTorch 모델을 ONNX 형식으로 변환합니다."""
    logger = logging.getLogger("triton_model_setup")
    
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

def generate_triton_config(config, model_name, output_path):
    """Triton 서버용 config.pbtxt 파일을 생성합니다."""
    logger = logging.getLogger("triton_model_setup")
    
    # 필수 설정 추출
    input_shape = config.get("INPUT_SHAPE", [96, 96, 96])
    in_channels = config.get("CHANNEL_IN", 1)
    out_channels = config.get("CHANNEL_OUT", 1)
    
    # config.pbtxt 파일 내용 생성
    config_content = f"""name: "{model_name}"
platform: "onnx"
max_batch_size: 1
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ {in_channels}, {input_shape[0]}, {input_shape[1]}, {input_shape[2]} ]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ {out_channels}, {input_shape[0]}, {input_shape[1]}, {input_shape[2]} ]
  }}
]
"""
    
    # 출력 디렉토리 확인
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 파일 저장
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Triton 설정 파일이 생성되었습니다: {output_path}")
    return output_path

def main():
    logger = setup_logging()
    
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description='Triton 서버용 모델 및 설정 준비')
    parser.add_argument('-c', '--config', type=str, required=True, help='설정 파일 이름 (쉼표로 구분 가능)')
    parser.add_argument('-o', '--output-dir', type=str, default='triton_model_repository', help='출력 디렉토리')
    parser.add_argument('-v', '--verbose', action='store_true', help='자세한 출력')
    
    args = parser.parse_args()
    
    # 쉼표로 구분된 모델 목록
    model_list = args.config.split(',')
    output_dir = Path(args.output_dir)
    
    for model_name in model_list:
        model_name = model_name.strip()
        logger.info(f"===== {model_name} 모델 처리 중 =====")
        
        try:
            # 모델 로드
            model, config = load_model(model_name)
            
            # 모델 디렉토리 설정
            model_dir = output_dir / model_name
            version_dir = model_dir / "1"
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # ONNX 변환
            onnx_path = version_dir / "model.onnx"
            convert_to_onnx(model, config, onnx_path, args.verbose)
            
            # Triton 설정 파일 생성
            config_path = model_dir / "config.pbtxt"
            generate_triton_config(config, model_name, config_path)
            
            logger.info(f"===== {model_name} 모델 처리 완료 =====")
        except Exception as e:
            logger.error(f"모델 {model_name} 처리 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("모든 모델 처리가 완료되었습니다.")
    logger.info(f"모델 저장소: {output_dir}")

if __name__ == "__main__":
    main()
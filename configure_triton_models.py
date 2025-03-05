#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json
import re
from pathlib import Path
import subprocess
import shutil

def setup_logging(log_level=logging.INFO):
    """로깅 설정"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("triton_model_configure")

def detect_gpu():
    """GPU 사용 가능 여부 감지"""
    try:
        subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def scan_model_directory(directory):
    """모델 디렉토리 스캔하여 사용 가능한 모델 목록 반환"""
    models = []
    directory = Path(directory)
    
    # 모델 디렉토리 확인
    if not directory.exists():
        return []
    
    # 각 서브디렉토리 확인
    for item in directory.iterdir():
        if item.is_dir():
            # 모델 버전 디렉토리(숫자명)와 config.pbtxt 확인
            if (item / "config.pbtxt").exists() or any((item / v).is_dir() for v in ["1", "2", "3"]):
                models.append(item.name)
    
    return models

def parse_config_pbtxt(config_path):
    """config.pbtxt 파일 파싱하여 모델 정보 추출"""
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 기본 정보 추출
    info = {}
    
    # 모델 이름
    name_match = re.search(r'name\s*:\s*"([^"]+)"', content)
    if name_match:
        info['name'] = name_match.group(1)
    
    # 플랫폼
    platform_match = re.search(r'platform\s*:\s*"([^"]+)"', content)
    if platform_match:
        info['platform'] = platform_match.group(1)
    
    # 최대 배치 크기
    batch_match = re.search(r'max_batch_size\s*:\s*(\d+)', content)
    if batch_match:
        info['max_batch_size'] = int(batch_match.group(1))
    
    # 입력 정보
    input_dims_match = re.search(r'input\s*\{[^}]*dims\s*:\s*\[\s*([^\]]+)\s*\]', content, re.DOTALL)
    if input_dims_match:
        dims_str = input_dims_match.group(1).strip()
        info['input_dims'] = [d.strip() for d in dims_str.split(',')]
    
    # 출력 정보
    output_dims_match = re.search(r'output\s*\{[^}]*dims\s*:\s*\[\s*([^\]]+)\s*\]', content, re.DOTALL)
    if output_dims_match:
        dims_str = output_dims_match.group(1).strip()
        info['output_dims'] = [d.strip() for d in dims_str.split(',')]
    
    return info

def update_config_for_cpu(config_path, model_info, backup=True):
    """config.pbtxt 파일을 CPU 모드로 업데이트"""
    logger = logging.getLogger("triton_model_configure")
    
    if not config_path.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return False
    
    # 백업 생성
    if backup:
        backup_path = config_path.with_suffix('.pbtxt.bak')
        shutil.copy2(config_path, backup_path)
        logger.info(f"설정 백업 생성: {backup_path}")
    
    # 새 설정 파일 생성
    new_config = f"""name: "{model_info['name']}"
platform: "{model_info['platform']}"
max_batch_size: {model_info.get('max_batch_size', 1)}
input {{
  name: "input"
  data_type: TYPE_FP32
  dims: [ {', '.join(model_info['input_dims'])} ]
}}
output {{
  name: "output"
  data_type: TYPE_FP32
  dims: [ {', '.join(model_info['output_dims'])} ]
}}

# CPU 실행을 위한 설정
instance_group {{
  count: 1
  kind: KIND_CPU
}}
"""
    
    with open(config_path, 'w') as f:
        f.write(new_config)
    
    logger.info(f"모델 {model_info['name']}의 설정 파일이 CPU 모드로 업데이트되었습니다.")
    return True

def update_config_for_gpu(config_path, model_info, backup=True):
    """config.pbtxt 파일을 GPU 모드로 업데이트"""
    logger = logging.getLogger("triton_model_configure")
    
    if not config_path.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return False
    
    # 백업 생성
    if backup:
        backup_path = config_path.with_suffix('.pbtxt.bak')
        shutil.copy2(config_path, backup_path)
        logger.info(f"설정 백업 생성: {backup_path}")
    
    # 새 설정 파일 생성
    new_config = f"""name: "{model_info['name']}"
platform: "{model_info['platform']}"
max_batch_size: {model_info.get('max_batch_size', 1)}
input {{
  name: "input"
  data_type: TYPE_FP32
  dims: [ {', '.join(model_info['input_dims'])} ]
}}
output {{
  name: "output"
  data_type: TYPE_FP32
  dims: [ {', '.join(model_info['output_dims'])} ]
}}

# GPU 실행을 위한 설정
instance_group {{
  count: 1
  kind: KIND_GPU
  gpus: [ 0 ]
}}
"""
    
    with open(config_path, 'w') as f:
        f.write(new_config)
    
    logger.info(f"모델 {model_info['name']}의 설정 파일이 GPU 모드로 업데이트되었습니다.")
    return True

def main():
    parser = argparse.ArgumentParser(description='Triton 모델 설정 관리 도구')
    parser.add_argument('-r', '--repository', type=str, default='triton_model_repository',
                       help='Triton 모델 저장소 경로')
    parser.add_argument('-m', '--models', type=str, default='',
                       help='처리할 모델 (쉼표로 구분, 비어있으면 모든 모델)')
    parser.add_argument('-g', '--gpu', type=str, choices=['auto', 'yes', 'no'], default='auto',
                       help='GPU 사용 여부 (auto: 자동 감지, yes: 사용, no: 사용 안함)')
    parser.add_argument('-b', '--backup', action='store_true', default=True,
                       help='설정 파일 백업 생성')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='자세한 로그 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    # GPU 감지
    use_gpu = args.gpu
    if use_gpu == 'auto':
        has_gpu = detect_gpu()
        use_gpu = 'yes' if has_gpu else 'no'
        logger.info(f"GPU 자동 감지: {'사용 가능' if has_gpu else '사용 불가'}")
    
    # 모델 저장소 경로
    repo_path = Path(args.repository)
    if not repo_path.exists():
        logger.error(f"모델 저장소를 찾을 수 없습니다: {repo_path}")
        return 1
    
    # 모델 목록 결정
    if args.models:
        models_to_process = args.models.split(',')
    else:
        models_to_process = scan_model_directory(repo_path)
        logger.info(f"발견된 모델: {', '.join(models_to_process)}")
    
    if not models_to_process:
        logger.error("처리할 모델이 없습니다.")
        return 1
    
    # 각 모델 처리
    for model_name in models_to_process:
        model_dir = repo_path / model_name
        config_path = model_dir / "config.pbtxt"
        
        if not config_path.exists():
            logger.warning(f"모델 {model_name}의 설정 파일을 찾을 수 없습니다: {config_path}")
            continue
        
        # 모델 정보 파싱
        model_info = parse_config_pbtxt(config_path)
        if not model_info:
            logger.warning(f"모델 {model_name}의 설정을 파싱할 수 없습니다.")
            continue
        
        # 설정 업데이트
        if use_gpu == 'yes':
            update_config_for_gpu(config_path, model_info, args.backup)
        else:
            update_config_for_cpu(config_path, model_info, args.backup)
    
    logger.info("모든 모델 설정 업데이트가 완료되었습니다.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3

import os
import logging
import argparse
import importlib
from pathlib import Path

from holoscan.core import Application
from holoscan.conditions import CountCondition

# 기존 연산자 가져오기
from operators.data_io_operator import (
    ImageLoaderOperator, 
    ImageSaverOperator,
    ResultDisplayOperator
)

# Triton 클라이언트 연산자 가져오기
from operators.triton_inference_operator import TritonInferenceOperator

class HoloscanTritonApp(Application):
    def __init__(self, *args, **kwargs):
        """애플리케이션 인스턴스 생성"""
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def run(self, *args, **kwargs):
        """애플리케이션 실행"""
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """애플리케이션별 연산자 생성 및 처리 그래프에 연결"""
        self._logger.debug(f"Begin {self.compose.__name__}")
        
        # 명령줄 인수 파싱
        parser = argparse.ArgumentParser(description='Holoscan-Triton 세그멘테이션 작업자')
        parser.add_argument('-i', '--input_file', type=str, help='입력 데이터 경로')
        parser.add_argument('-o', '--output_dir', type=str, help='출력 데이터 경로')
        parser.add_argument('-c', '--configs', type=str, help='설정 파일명')
        parser.add_argument('-t', '--triton-url', type=str, default='localhost:8000', help='Triton 서버 URL')
        parser.add_argument('-m', '--model-name', type=str, default='monai_segmentation', help='모델 이름')
        
        args, unknown = parser.parse_known_args(self.argv)
        
        # 경로 설정
        input_path = Path(args.input_file) 
        output_path = Path(args.output_dir) 
        config_name = args.configs 
        triton_url = args.triton_url
        model_name = args.model_name
        
        self._logger.info(f"입력 경로: {input_path}, 출력 경로: {output_path}, 설정: {config_name}")
        self._logger.info(f"Triton URL: {triton_url}, 모델 이름: {model_name}")
        
        # 설정 가져오기
        try:
            _module = importlib.import_module(f"config.{config_name}")
            config = _module.config
            transform = _module.transform
        except ImportError:
            self._logger.error(f"설정 모듈을 가져올 수 없습니다: config.{config_name}")
            raise
        
        # 연산자 생성
        image_loader_op = ImageLoaderOperator(
            self, 
            config=config,
            input_path=input_path,
            name="image_loader_op"
        )
        
        # Triton 추론 연산자 (PyTorch 직접 추론 대신 사용)
        triton_inference_op = TritonInferenceOperator(
            self,
            triton_url=triton_url,
            model_name=model_name,
            config=config,
            name="triton_inference_op"
        )

        # 결과 표시 연산자
        result_display_op = ResultDisplayOperator(
            self,
            display_interval=1.0,
            name="result_display_op"
        )
        
        # 결과 저장 연산자
        image_saver_op = ImageSaverOperator(
            self,
            output_dir=output_path,
            name="image_saver_op"
        )
        
        # 연산자를 흐름에 연결
        self.add_flow(image_loader_op, triton_inference_op, {(image_loader_op.output_name, "image")})
        self.add_flow(triton_inference_op, result_display_op, {("prediction", result_display_op.input_name)})
        self.add_flow(triton_inference_op, image_saver_op, {("prediction", image_saver_op.input_name)})
        
        self._logger.debug(f"End {self.compose.__name__}")


if __name__ == "__main__":
    # 기본 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Begin {__name__}")
    
    # 애플리케이션 생성 및 실행
    HoloscanTritonApp().run()
    
    logging.info(f"End {__name__}")
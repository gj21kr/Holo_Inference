import os
import logging
import numpy as np
import torch
from pathlib import Path

from holoscan.core import Operator, OperatorSpec, Fragment
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

class TritonInferenceOperator(Operator):
    """Triton Inference Server 클라이언트 연산자
    
    이 연산자는 Holoscan 파이프라인에서 Triton Inference Server로 추론 요청을 보내는 역할을 합니다.
    전처리된 이미지를 입력으로 받아 Triton 서버로 전송하고, 추론 결과를 후속 연산자에게 제공합니다.
    """
    
    def __init__(
        self,
        fragment: Fragment,
        *args,
        triton_url: str = "localhost:8000",
        model_name: str = "monai_segmentation",
        model_version: str = "1",
        timeout: int = 60000,
        config=None,
        **kwargs
    ):
        """
        Args:
            fragment: Holoscan Fragment 객체
            triton_url: Triton 서버 URL (기본값: "localhost:8000")
            model_name: 추론에 사용할 모델 이름 (기본값: "monai_segmentation")
            model_version: 모델 버전 (기본값: "1")
            timeout: 추론 요청 타임아웃(ms) (기본값: 60000)
            config: 모델 설정 (CHANNEL_OUT, INPUT_SHAPE 등)
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.timeout = timeout
        self.config = config
        
        self.input_name = "input"
        self.output_name = "output"
        
        # 로거 설정
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        
        super().__init__(fragment, *args, **kwargs)
    
    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.output("prediction")
    
    def compute(self, op_input, op_output, context):
        """
        입력 이미지에 대해 Triton을 통한 추론을 수행하고 결과를 출력합니다.
        
        Args:
            op_input: Holoscan 입력 컨텍스트
            op_output: Holoscan 출력 컨텍스트
            context: Holoscan 실행 컨텍스트
        """
        # 입력 데이터 받기
        input_data = op_input.receive("image")
        if input_data is None:
            raise ValueError("입력 이미지가 없습니다.")
        
        # 입력 이미지 처리
        image = input_data.get("image", None)
        meta = input_data.get("meta", {})
        
        # 입력이 PyTorch 텐서인 경우 NumPy 배열로 변환
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Triton 클라이언트 생성
        try:
            client = httpclient.InferenceServerClient(
                url=self.triton_url, verbose=False
            )
        except Exception as e:
            self._logger.error(f"Triton 서버에 연결할 수 없습니다: {e}")
            raise
        
        # 입력 데이터 형식 변환
        if len(image.shape) == 3:  # 채널 차원 추가
            image = np.expand_dims(image, axis=0)
        
        if len(image.shape) == 4 and image.shape[0] != 1:  # 배치 차원 추가
            image = np.expand_dims(image, axis=0)
        
        # 입력 텐서 설정
        inputs = []
        inputs.append(httpclient.InferInput(self.input_name, image.shape, "FP32"))
        inputs[0].set_data_from_numpy(image.astype(np.float32))
        
        # 출력 텐서 설정
        outputs = []
        outputs.append(httpclient.InferRequestedOutput(self.output_name))
        
        # 추론 요청 보내기
        self._logger.info(f"Triton 서버로 추론 요청 전송 중 ({self.model_name})...")
        
        try:
            response = client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
                client_timeout=self.timeout
            )
        except Exception as e:
            self._logger.error(f"추론 중 오류 발생: {e}")
            raise
        
        # 추론 결과 처리
        result = response.as_numpy(self.output_name)
        
        # 채널 차원이 있다면 처리
        if len(result.shape) > 3 and result.shape[0] == 1:
            result = np.squeeze(result, axis=0)
        
        # 활성화 함수 적용 (필요한 경우)
        if self.config and "ACTIVATION" in self.config:
            if self.config["ACTIVATION"].lower() == "sigmoid":
                result = 1.0 / (1.0 + np.exp(-result))
            elif self.config["ACTIVATION"].lower() == "softmax":
                # softmax 구현 (채널 축으로)
                exp_result = np.exp(result - np.max(result, axis=0, keepdims=True))
                result = exp_result / np.sum(exp_result, axis=0, keepdims=True)
        
        # 임계값 적용 (필요한 경우)
        if self.config and "THRESHOLD" in self.config:
            result = (result > self.config["THRESHOLD"]).astype(np.uint8)
        
        # 결과 출력
        output_data = {
            "image": result,
            "meta": meta,
        }
        
        op_output.emit(output_data, "prediction")
        self._logger.info(f"추론 완료 (결과 shape: {result.shape})")
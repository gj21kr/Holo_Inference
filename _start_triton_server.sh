#!/bin/bash
# Triton 서버 시작 스크립트 - 유연한 설정

# 기본 설정 변수 (환경 변수로 변경 가능)
MODEL_REPOSITORY=${MODEL_REPOSITORY:-"/home/holoscan/Documents/Holo_Inference/triton_model_repository"}
TRITON_IMAGE=${TRITON_IMAGE:-"nvcr.io/nvidia/tritonserver:23.10-py3"}
CONTAINER_NAME=${CONTAINER_NAME:-"triton_server"}
USE_GPU=${USE_GPU:-"auto"}  # auto, yes, no
CUDA_MEM_POOL=${CUDA_MEM_POOL:-"0:536870912"}  # GPU_ID:SIZE (512MB)
PINNED_MEM_POOL=${PINNED_MEM_POOL:-"536870912"}  # 512MB
LOG_LEVEL=${LOG_LEVEL:-"1"}
MODEL_CONTROL_MODE=${MODEL_CONTROL_MODE:-"none"}  # none, poll, explicit
PORT_HTTP=${PORT_HTTP:-"8000"}
PORT_GRPC=${PORT_GRPC:-"8001"}
PORT_METRICS=${PORT_METRICS:-"8002"}
MODELS_TO_LOAD=${MODELS_TO_LOAD:-""}  # 쉼표로 구분된 모델 목록 (비어있으면 모든 모델)

# 명령줄 인자 파싱
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-repository=*)
      MODEL_REPOSITORY="${1#*=}"
      ;;
    --use-gpu=*)
      USE_GPU="${1#*=}"
      ;;
    --image=*)
      TRITON_IMAGE="${1#*=}"
      ;;
    --container-name=*)
      CONTAINER_NAME="${1#*=}"
      ;;
    --log-level=*)
      LOG_LEVEL="${1#*=}"
      ;;
    --models=*)
      MODELS_TO_LOAD="${1#*=}"
      ;;
    --cuda-mem=*)
      CUDA_MEM_POOL="${1#*=}"
      ;;
    --pinned-mem=*)
      PINNED_MEM_POOL="${1#*=}"
      ;;
    --mode=*)
      MODEL_CONTROL_MODE="${1#*=}"
      ;;
    --help)
      echo "사용법: $0 [옵션]"
      echo "옵션:"
      echo "  --model-repository=PATH   모델 저장소 경로 (기본값: $MODEL_REPOSITORY)"
      echo "  --use-gpu=yes|no|auto     GPU 사용 여부 (기본값: auto)"
      echo "  --image=IMAGE             Triton 서버 이미지 (기본값: $TRITON_IMAGE)"
      echo "  --container-name=NAME     컨테이너 이름 (기본값: $CONTAINER_NAME)"
      echo "  --log-level=LEVEL         로그 수준 (기본값: $LOG_LEVEL)"
      echo "  --models=MODEL1,MODEL2    로드할 모델 목록 (기본값: 모든 모델)"
      echo "  --cuda-mem=ID:SIZE        CUDA 메모리 풀 크기 (기본값: $CUDA_MEM_POOL)"
      echo "  --pinned-mem=SIZE         고정 메모리 풀 크기 (기본값: $PINNED_MEM_POOL)"
      echo "  --mode=none|poll|explicit 모델 컨트롤 모드 (기본값: $MODEL_CONTROL_MODE)"
      echo "  --help                    이 도움말 표시"
      exit 0
      ;;
    *)
      echo "알 수 없는 옵션: $1"
      echo "$0 --help 로 도움말을 확인하세요."
      exit 1
      ;;
  esac
  shift
done

# 이전 실행 중인 컨테이너 정리
echo "이전 Triton 서버 컨테이너 정리 중..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# 모델 저장소 확인
if [ ! -d "$MODEL_REPOSITORY" ]; then
  echo "오류: 모델 저장소를 찾을 수 없습니다: $MODEL_REPOSITORY"
  exit 1
fi

# 모델 저장소 내용 출력
echo "============================="
echo "모델 저장소 내용 확인:"
ls -la $MODEL_REPOSITORY
echo "모델 디렉토리 확인:"
find $MODEL_REPOSITORY -type d | grep -v "__pycache__"
echo "모델 파일 확인:"
find $MODEL_REPOSITORY -name "*.onnx" -o -name "config.pbtxt"
echo "============================="

# GPU 상태 확인 및 자동 감지
if [ "$USE_GPU" = "auto" ]; then
  if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU 감지됨, GPU 모드로 실행합니다."
    USE_GPU="yes"
  else
    echo "GPU를 감지할 수 없음, CPU 모드로 실행합니다."
    USE_GPU="no"
  fi
fi

# Docker 실행 옵션 설정
DOCKER_RUN_OPTS="-d -p $PORT_HTTP:8000 -p $PORT_GRPC:8001 -p $PORT_METRICS:8002 --name $CONTAINER_NAME"
if [ "$USE_GPU" = "yes" ]; then
  DOCKER_RUN_OPTS="$DOCKER_RUN_OPTS --runtime=nvidia"
  echo "GPU 모드로 실행합니다."
else
  echo "CPU 모드로 실행합니다."
fi

# Docker 컨테이너 시작
echo "Triton 서버 컨테이너 시작 중..."
docker run $DOCKER_RUN_OPTS $TRITON_IMAGE sleep infinity

# 모델 파일 복사
echo "모델 파일 복사 중..."
docker exec $CONTAINER_NAME mkdir -p /models
docker cp $MODEL_REPOSITORY/. $CONTAINER_NAME:/models/

# 모델 디렉토리 권한 설정
echo "모델 디렉토리 권한 설정 중..."
docker exec $CONTAINER_NAME chmod -R 755 /models

# Triton 서버 시작 명령 구성
TRITON_CMD="tritonserver --model-repository=/models --log-verbose=$LOG_LEVEL --strict-model-config=false"

# 모델 컨트롤 모드 설정
if [ "$MODEL_CONTROL_MODE" != "none" ]; then
  TRITON_CMD="$TRITON_CMD --model-control-mode=$MODEL_CONTROL_MODE"
fi

# 지정된 모델만 로드
if [ -n "$MODELS_TO_LOAD" ] && [ "$MODEL_CONTROL_MODE" = "explicit" ]; then
  for MODEL in ${MODELS_TO_LOAD//,/ }; do
    TRITON_CMD="$TRITON_CMD --load-model=$MODEL"
  done
fi

# GPU 관련 설정 추가
if [ "$USE_GPU" = "yes" ]; then
  TRITON_CMD="$TRITON_CMD --cuda-memory-pool-byte-size=$CUDA_MEM_POOL --pinned-memory-pool-byte-size=$PINNED_MEM_POOL"
fi

# Triton 서버 시작
echo "Triton 서버 시작 중..."
echo "실행 명령: $TRITON_CMD"
docker exec -it $CONTAINER_NAME bash -c "$TRITON_CMD"

# # 스크립트가 여기까지 도달하면 서버 종료
# echo "Triton 서버가 종료되었습니다."
# docker stop $CONTAINER_NAME
# docker rm $CONTAINER_NAME
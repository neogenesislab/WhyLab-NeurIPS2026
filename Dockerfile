# ── WhyLab: 인과추론 연구 파이프라인 ──
# NVIDIA CUDA 12.2 + Python 3.11 + LightGBM GPU
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 비대화형 + 타임존
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git curl build-essential cmake \
    libboost-dev libboost-system-dev libboost-filesystem-dev \
    ocl-icd-opencl-dev opencl-headers \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화 + 심볼릭 링크
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /whylab

# 의존성 먼저 (Docker 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LightGBM GPU 빌드 (CUDA 지원)
RUN pip install --no-cache-dir lightgbm --config-settings=cmake.define.USE_GPU=ON

# 소스 코드 복사
COPY . .

# 환경 변수
ENV PYTHONPATH=/whylab

# 기본 엔트리포인트: 벤치마크 실행
ENTRYPOINT ["python", "-m", "engine"]

# 기본 인자: 도움말
CMD ["--help"]

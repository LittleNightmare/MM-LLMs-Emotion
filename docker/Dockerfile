# 平台镜像：基础镜像由平台按“资源配置 + 基础镜像”自动生成 FROM 行，
# 这里的 FROM 只是占位，本地想手工构建时才用得上。
#
# 平台现有基础镜像（2026-10-27 核对）：
#   开发机  CUDA 12.6 / PyTorch 2.4.0 / Python 3.11
#   训练    CUDA 12.8 / PyTorch 2.7.1 / Python 3.11
ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE_IMAGE}

# 国内源 + 不缓存；HF 走镜像，下载基座权重时省事
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# ffmpeg 给音视频兜底（av / imageio-ffmpeg 自带二进制，系统装一份更稳）
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# torch：开发机基础镜像是 2.4.0，transformers 5.20.2 要求 torch>=2.6，太旧。
# 统一装 2.7.1。PyPI 上 Linux 的 torch 轮子自带 CUDA 12.6（依赖 nvidia-*-cu12），
# 不是 CPU 版；装完立刻断言 torch.version.cuda，装成 CPU 版会直接构建失败。
# 要走别的 CUDA 版本就换成官方索引：
#   RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
#         torch==2.7.1+cu128 torchvision==0.22.1+cu128
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
 && python -c "import torch; assert torch.version.cuda, 'CPU-only torch build'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# 项目依赖，版本跟仓库 pyproject.toml 对齐；改依赖时两处一起改。
# 注意 av / librosa 必须是 3.11 兼容的版本（19.x 和 1.0 起要求 Python>=3.12）。
RUN pip install --no-cache-dir \
      "transformers==5.20.2" \
      "accelerate==1.15.0" \
      "huggingface_hub==2.2.0" \
      "av==18.1.0" \
      "numpy==2.3.5" \
      "pillow==12.0.0" \
      "soundfile==0.14.0" \
      "librosa==0.10.2" \
      "imageio-ffmpeg==0.6.0" \
      "scikit-learn==1.9.1" \
      "modelscope==1.40.2" \
      "kaggle==2.2.5"

# 自检：缺依赖时构建阶段就失败，而不是等到跑推理
RUN python -c "import torch, torchvision, transformers, av, librosa, sklearn, kaggle; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'torchvision', torchvision.__version__, 'transformers', transformers.__version__)"

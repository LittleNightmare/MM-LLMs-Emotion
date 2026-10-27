# 平台镜像：基础镜像由平台按“资源配置 + 基础镜像”自动生成 FROM 行，
# 这里的 FROM 只是占位，本地想手工构建时才用得上。
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

# torch / torchvision 用基础镜像自带的版本；这里只补项目依赖。
# 版本跟仓库 pyproject.toml 对齐，改依赖时两处一起改。
RUN pip install --no-cache-dir \
      "transformers==5.20.2" \
      "accelerate==1.15.0" \
      "huggingface_hub==2.2.0" \
      "av==19.0.1" \
      "numpy==2.3.5" \
      "pillow==12.0.0" \
      "soundfile==0.14.0" \
      "librosa==1.0.0" \
      "imageio-ffmpeg==0.6.0" \
      "scikit-learn==1.9.1" \
      "modelscope==1.40.2" \
      "kaggle==2.2.5"

# 自检：缺依赖时构建阶段就失败，而不是等到跑推理
RUN python -c "import torch, transformers, av, librosa, sklearn, kaggle; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"

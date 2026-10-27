#!/usr/bin/env bash
# 在 Intern Discovery 开发机里把仓库跑起来（Linux，不需要 sudo）。
#
# 用法（在仓库根目录）：
#   bash scripts/machine_bootstrap.sh                 # 只建目录、接线、装环境、体检
#   bash scripts/machine_bootstrap.sh --data          # 顺带下载竞赛数据（1.57 GB）
#   bash scripts/machine_bootstrap.sh --model qwen-omni-3b   # 顺带下载基座权重（约 12 GB）
#   bash scripts/machine_bootstrap.sh --no-uv        # 依赖已在镜像里，跳过 uv
#
# 约定：大件放 $ASSETS（默认 /data/assets），仓库里只留软链接；
# venv 和 uv 缓存也放 /data，避免开发机重置后重装。

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS="/data/assets"
DATASET_NAME="course-dataset"
ENV_DIR="/data/env/mm-llms-emotion"
UV_CACHE="/data/cache/uv"
WITH_DATA=0
MODEL=""
NO_UV=0

while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --data) WITH_DATA=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --no-uv) NO_UV=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

run_py() {
  if [ "$NO_UV" = "1" ]; then
    python "$@"
  else
    uv run python "$@"
  fi
}

say() { printf '\n== %s\n' "$*"; }

if [ ! -d /data ]; then
  echo "警告: /data 不存在，这里可能不是开发机；照常执行但文件不会持久化。"
fi

say "1/6 建目录"
mkdir -p "$ASSETS" "$ENV_DIR" "$(dirname "$UV_CACHE")"
# 云盘里传的是 course-dataset.tar（1572 MB），第一次跑就顺手解开
if [ ! -d "$ASSETS/$DATASET_NAME" ] && [ -f "$ASSETS/course-dataset.tar" ]; then
  echo "发现 $ASSETS/course-dataset.tar，解包中（约 1.5 GB）"
  tar -xf "$ASSETS/course-dataset.tar" -C "$ASSETS"
  if [ -d "$ASSETS/Dataset" ]; then
    mv "$ASSETS/Dataset" "$ASSETS/$DATASET_NAME"
  fi
  rm -f "$ASSETS/course-dataset.tar"
fi
mkdir -p "$ASSETS/$DATASET_NAME" "$ASSETS/models"
# 模型同样是云盘上的 tar，第一次跑就解开
if [ ! -d "$ASSETS/models/qwen-omni-3b" ] && [ -f "$ASSETS/models/qwen-omni-3b.tar" ]; then
  echo "发现 $ASSETS/models/qwen-omni-3b.tar，解包中（约 12 GB）"
  tar -xf "$ASSETS/models/qwen-omni-3b.tar" -C "$ASSETS/models"
  rm -f "$ASSETS/models/qwen-omni-3b.tar"
fi
echo "assets:   $ASSETS"
echo "dataset:  $ASSETS/$DATASET_NAME"
echo "env:      $ENV_DIR"

say "2/6 把 Dataset/ 和 models/ 接到 assets"
# 云盘上传时目录名还是 Dataset；第一次跑脚本就顺手改成不和平台 /dataset 混淆的名字。
if [ ! -d "$ASSETS/$DATASET_NAME" ] && [ -d "$ASSETS/Dataset" ]; then
  mv "$ASSETS/Dataset" "$ASSETS/$DATASET_NAME"
  echo "已把 $ASSETS/Dataset 改名为 $ASSETS/$DATASET_NAME"
fi
if [ -L "$REPO/Dataset" ]; then
  ln -sfn "$ASSETS/$DATASET_NAME" "$REPO/Dataset"
elif [ -d "$REPO/Dataset" ]; then
  echo "保留已有目录 $REPO/Dataset（不是软链接，跳过）"
else
  ln -s "$ASSETS/$DATASET_NAME" "$REPO/Dataset"
fi
if [ -L "$REPO/models" ]; then
  ln -sfn "$ASSETS/models" "$REPO/models"
elif [ -d "$REPO/models" ]; then
  echo "保留已有目录 $REPO/models（不是软链接，跳过）"
else
  ln -s "$ASSETS/models" "$REPO/models"
fi

say "3/6 准备 Python 环境"
cd "$REPO"
if [ "$NO_UV" = "1" ]; then
  echo "跳过 uv，使用镜像里的 Python"
  python -V
else
  export UV_CACHE_DIR="$UV_CACHE"
  export UV_PROJECT_ENVIRONMENT="$ENV_DIR"
  if ! command -v uv >/dev/null 2>&1; then
    echo "没找到 uv，用 pip 装一个到用户目录"
    python -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple uv
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv --version
fi

say "4/6 同步依赖"
if [ "$NO_UV" = "1" ]; then
  python -c "import importlib.metadata as m, torch, transformers, av, librosa, sklearn; print('依赖齐全, kaggle CLI', m.version('kaggle'))"
else
  echo "首次会下 Python 3.12 和 CUDA 版 torch，几分钟"
  uv sync
fi

say "5/6 拉资源"
if [ "$WITH_DATA" = "1" ]; then
  run_py scripts/fetch_assets.py dataset
else
  echo "跳过数据（--data 才下载）"
fi
if [ -n "$MODEL" ]; then
  run_py scripts/fetch_assets.py models "$MODEL"
else
  echo "跳过权重（--model qwen-omni-3b 才下载）"
fi

say "6/6 体检"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "没有 nvidia-smi：这台是纯 CPU 资源？"
fi
run_py - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0),
          "%.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
PY
df -h /data | tail -n 1
for url in https://www.kaggle.com https://hf-mirror.com https://pypi.tuna.tsinghua.edu.cn; do
  code=$(curl -m 8 -s -o /dev/null -w '%{http_code}' "$url" || echo "fail")
  echo "网络 $url -> $code"
done

say "完成"
echo "仓库:   $REPO"
echo "数据:   $ASSETS/$DATASET_NAME"
echo "权重:   $ASSETS/models"
echo "venv:   $ENV_DIR"
if [ "$NO_UV" = "1" ]; then
  echo "模式: 镜像自带依赖（--no-uv），直接用 python"
else
  echo "之后每次进机器都要先: export UV_PROJECT_ENVIRONMENT=$ENV_DIR UV_CACHE_DIR=$UV_CACHE"
fi

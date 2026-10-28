#!/usr/bin/env bash
# 把 LoRA adapter 合并成完整权重，供推理镜像里的 baseline.py 评测。
#
#   bash scripts/export_merged.sh /data/runs/sft-s42/checkpoint-800 /data/tmp/merged-sft-42
#
# 合并后 12 GB 左右，评测完立刻删；云盘放不下多份。
set -euo pipefail

CKPT=${1:?用法: export_merged.sh <checkpoint 目录> [输出目录]}
OUT=${2:-/data/tmp/merged-$(date +%H%M%S)}
BASE=${BASE:-/data/assets/models/qwen-omni-3b}

swift export \
  --model "$BASE" \
  --adapters "$CKPT" \
  --merge_lora true \
  --output_dir "$OUT"

du -sh "$OUT"
echo "合并完成：$OUT"

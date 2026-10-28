#!/usr/bin/env bash
# 冒烟：10 条 SFT，1 epoch。只验证「能跑、不 OOM、模板能吃视频/音频」。
#
# VT/VA 分开探针：
#   DATASET=/data/assets/swift/probe_vt.jsonl OUT=/data/runs/probe-vt bash scripts/smoke_swift_sft.sh
#   DATASET=/data/assets/swift/probe_va.jsonl OUT=/data/runs/probe-va bash scripts/smoke_swift_sft.sh
set -euo pipefail

ln -sfn /data/assets/course-dataset /code/workspace/Dataset 2>/dev/null || true
ln -sfn /data/assets/models /code/workspace/models 2>/dev/null || true
export USE_AUDIO_IN_VIDEO=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT=${OUT:-/data/runs/smoke-sft}
DATASET=${DATASET:-/data/assets/swift/sft_smoke.jsonl}

swift sft \
  --model /data/assets/models/qwen-omni-3b --template qwen2_omni \
  --dataset "$DATASET" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 --max_length 4096 \
  --output_dir "$OUT" --logging_steps 1 --save_steps 10 --save_total_limit 1

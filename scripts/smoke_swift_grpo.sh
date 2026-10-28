#!/usr/bin/env bash
# 冒烟：2 步 GRPO，G=2，打开 completion 日志，验证奖励插件被调用。
set -euo pipefail

ln -sfn /data/assets/course-dataset /code/workspace/Dataset 2>/dev/null || true
ln -sfn /data/assets/models /code/workspace/models 2>/dev/null || true
export USE_AUDIO_IN_VIDEO=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT=${OUT:-/data/runs/smoke-grpo}
DATASET=${DATASET:-/data/assets/swift/grpo_smoke.jsonl}

swift rlhf --rlhf_type grpo \
  --model /data/assets/models/qwen-omni-3b --template qwen2_omni \
  --dataset "$DATASET" \
  --external_plugins /code/workspace/scripts/swift_reward_plugin.py \
  --reward_funcs emotion_conflict_format_overask \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --num_generations 2 --max_completion_length 192 --temperature 1.0 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 --max_steps 2 --beta 0.02 \
  --output_dir "$OUT" --logging_steps 1 --save_steps 2 --save_total_limit 1 \
  --log_completions true \
  --report_to tensorboard

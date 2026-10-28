#!/usr/bin/env bash
# 训练任务的 GRPO 启动脚本（ms-swift + 自定义奖励插件）。
#
# 从 SFT adapter 初始化：SFT_ADAPTER=/data/runs/sft-swift/checkpoint-xxx
# 不设 SFT_ADAPTER 就从基座直接 RL（只用于冒烟对比）。
set -euo pipefail

ln -sfn /data/assets/course-dataset /code/workspace/Dataset 2>/dev/null || true
ln -sfn /data/assets/models /code/workspace/models 2>/dev/null || true
export USE_AUDIO_IN_VIDEO=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT=${OUT:-/data/runs/grpo-swift}
STEPS=${STEPS:-250}
GEN=${GEN:-4}
SEED=${SEED:-42}
FUNCS=${REWARD_FUNCS:-emotion_conflict_format_overask}
WEIGHTS=${REWARD_WEIGHTS:-}
EXTRA=()
if [ -n "${SFT_ADAPTER:-}" ]; then
  EXTRA=(--adapters "$SFT_ADAPTER")
fi
REWARD_EXTRA=()
if [ -n "$WEIGHTS" ]; then
  REWARD_EXTRA=(--reward_weights $WEIGHTS)
fi

swift rlhf --rlhf_type grpo \
  --model /data/assets/models/qwen-omni-3b --template qwen2_omni \
  "${EXTRA[@]}" \
  --dataset /data/assets/swift/grpo_train.jsonl \
  --external_plugins /code/workspace/scripts/swift_reward_plugin.py \
  --reward_funcs $FUNCS "${REWARD_EXTRA[@]}" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --num_generations "$GEN" --max_completion_length 192 --temperature 1.0 --seed "$SEED" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 --max_steps "$STEPS" --beta 0.02 \
  --output_dir "$OUT" --logging_steps 1 --save_steps 100 --save_total_limit 2 \
  --report_to tensorboard

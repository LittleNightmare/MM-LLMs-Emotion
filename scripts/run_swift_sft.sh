#!/usr/bin/env bash
# 训练任务的 SFT 启动脚本（ms-swift）。
#
# 训练任务从 Git 拉代码时，/code/workspace 下没有 Dataset/models 软链，先补上。
# 数据与权重都在云盘：/data/assets/course-dataset、/data/assets/models/qwen-omni-3b。
# 依赖与验证步骤见 docs/训练镜像-ms-swift.md。
set -euo pipefail

ln -sfn /data/assets/course-dataset /code/workspace/Dataset 2>/dev/null || true
ln -sfn /data/assets/models /code/workspace/models 2>/dev/null || true
export USE_AUDIO_IN_VIDEO=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT=${OUT:-/data/runs/sft-swift}
DATASET=${DATASET:-/data/assets/swift/sft_train.jsonl}
SEED=${SEED:-42}

swift sft \
  --model /data/assets/models/qwen-omni-3b --template qwen2_omni \
  --dataset "$DATASET" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 --num_train_epochs 1 --seed "$SEED" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 --max_length 4096 \
  --output_dir "$OUT" --logging_steps 5 --save_steps 200 --save_total_limit 2 \
  --dataloader_num_workers 2 --report_to tensorboard

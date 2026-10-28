#!/usr/bin/env bash
# 一次跑完四条冒烟：SFT 10 条、VT 探针、VA 探针、GRPO 2 步。
# 任何一条失败即停，后面的不再跑。
set -euo pipefail

ROOT=/code/workspace
export USE_AUDIO_IN_VIDEO=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "== 1/4 SFT 10 条 =="
DATASET=/data/assets/swift/sft_smoke.jsonl OUT=/data/runs/smoke-sft \
  bash "$ROOT/scripts/smoke_swift_sft.sh"

echo "== 2/4 VT 探针 1 条 =="
DATASET=/data/assets/swift/probe_vt.jsonl OUT=/data/runs/probe-vt \
  bash "$ROOT/scripts/smoke_swift_sft.sh"

echo "== 3/4 VA 探针 1 条 =="
DATASET=/data/assets/swift/probe_va.jsonl OUT=/data/runs/probe-va \
  bash "$ROOT/scripts/smoke_swift_sft.sh"

echo "== 4/4 GRPO 2 步（带 completion 日志）=="
DATASET=/data/assets/swift/grpo_smoke.jsonl OUT=/data/runs/smoke-grpo \
  bash "$ROOT/scripts/smoke_swift_grpo.sh"

echo "== 冒烟结束：检查 /data/runs/smoke-sft、probe-vt、probe-va、smoke-grpo 的日志 =="

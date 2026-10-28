# 训练镜像（ms-swift）

记录时间：2026-10-28。目标：给 Intern Discovery 的**训练任务**建一个能跑 Qwen2.5-Omni SFT / GRPO 的镜像；推理镜像继续钉 `transformers==5.20.2`，两边互不影响。

训练框架选型与理由见 [训练方案](训练方案.md) 第 3 节；框架是 [ms-swift](https://github.com/modelscope/ms-swift)，它支持 Qwen2.5-Omni 的 SFT/DPO/GRPO（[PR #3613](https://github.com/modelscope/ms-swift/pull/3613)），GRPO 家族全、LoRA、自定义奖励插件。

## 1. 建镜像

平台「镜像」页新建，使用场景选**训练任务**（不要选开发机，两个场景镜像分开），资源配置 NvidiaA100-1-80G，基础镜像挑 PyTorch 2.7.1 / CUDA 12.8 / Python 3.11 那一档。

两种装法任选：

- **快捷安装**：apt 填 `ffmpeg git curl ca-certificates`；pip 抄 [docker/pip-packages-train.txt](../docker/pip-packages-train.txt)。
- **Dockerfile**：用 [docker/Dockerfile.train](../docker/Dockerfile.train)（平台自动填 FROM）。

要点：

- 不要在这个镜像里钉 `transformers==5.20.2`。ms-swift 要自己的版本组合（官方推荐 transformers 5.12.x、trl 0.29、peft<0.21），钉死会解析冲突。
- `decord`、`qwen_omni_utils` 是 Qwen2.5-Omni 读视频的依赖，`soundfile` 读音频，缺一个都会在模板编码阶段报错。
- 镜像里已经有 `ffmpeg`，`to_swift.py` 抽音轨时也会用 `imageio-ffmpeg` 的静态二进制。

## 2. 构建后自检

进开发机或训练任务的 shell，先跑：

```bash
python -c "import torch, swift, decord, soundfile; print(torch.__version__, torch.version.cuda, swift.__version__)"
python -c "from swift.rewards import ORM; print('reward api ok')"
swift --help | head -30
nvidia-smi --query-gpu=name,memory.total --format=csv
```

期望：torch 报 CUDA 可用；`swift --help` 能列出 `sft` / `rlhf` 子命令；GPU 是 A100-80G。

## 3. 准备数据

数据转换在开发机上跑（推理镜像就能做，不需要 ms-swift）：

```bash
# SFT：带 assistant 目标 JSON
python scripts/to_swift.py --task sft --dataset Dataset/train.jsonl \
    --output /data/assets/swift/sft_train.jsonl

# GRPO：只要 prompt + 奖励列
python scripts/to_swift.py --task grpo --dataset Dataset/train.jsonl \
    --output /data/assets/swift/grpo_train.jsonl
```

VA 行会把音轨抽成 16 kHz 单声道 wav 放在 `/data/assets/swift/audio/`（约 1–2 GB，按 VA 条数估算）。转换结束会打印统计：`VT / VA / 缺失媒体 / 静音兜底 / 有补标` 的条数，缺失媒体的行被跳过。

有补标文件时加 `--annotations 补标.jsonl`（字段 `sample_id, is_uncertain, clarifying_question`）；没有补标也能转，`needs_clarification` 为 null，追问奖励项跳过。

## 4. 冒烟（先做，再全量）

```bash
# 10 条 SFT
python scripts/to_swift.py --task sft --dataset Dataset/train.jsonl --limit 10 \
    --output /data/assets/swift/sft_smoke.jsonl

swift sft \
  --model /data/assets/models/qwen-omni-3b --template qwen2_omni \
  --dataset /data/assets/swift/sft_smoke.jsonl \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 --num_train_epochs 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 --max_length 4096 \
  --output_dir /data/runs/smoke-sft --logging_steps 1 --save_steps 10 --save_total_limit 1
```

```bash
# 2 步 GRPO，带奖励插件
python scripts/to_swift.py --task grpo --dataset Dataset/train.jsonl --limit 10 \
    --output /data/assets/swift/grpo_smoke.jsonl

swift rlhf --rlhf_type grpo \
  --model /data/assets/models/qwen-omni-3b --template qwen2_omni \
  --dataset /data/assets/swift/grpo_smoke.jsonl \
  --external_plugins scripts/swift_reward_plugin.py \
  --reward_funcs emotion_conflict_format_overask \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --num_generations 4 --max_completion_length 192 --temperature 1.0 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 --max_steps 2 \
  --output_dir /data/runs/smoke-grpo --logging_steps 1 --save_steps 2 --save_total_limit 1
```

参数名以 `swift rlhf --help` 为准（`--beta`、`--max_completion_length` 这类在版本间可能改名）。

## 5. 冒烟判据

1. SFT smoke 能跑完，loss 有下降，不 OOM。
2. GRPO smoke 的日志里能看到每个 completion 的 reward 分（四维里至少情感/冲突/格式三项有值），completion 不是空串或全截断。
3. **VT 不泄漏音频**：VT 行的渲染输入里不应出现 audio token；VA 行应该有。看 `--log_completions` 或模板日志，必要时用一条 VT、一条 VA 单独跑。
4. adapter 可回传：`swift export --adapters /data/runs/smoke-sft/checkpoint-10 --merge_lora true --output_dir /data/tmp/merged-smoke`，然后在推理镜像里用 `baseline.py --model qwen-omni-3b --model-path /data/tmp/merged-smoke --limit 5` 跑通。

四条都过再上全量；第 3 条不过就先调整 `to_swift.py` 的占位符或 `USE_AUDIO_IN_VIDEO`，不要带着音频泄漏训练。

## 6. 正式运行

SFT 全量（换 `--dataset /data/assets/swift/sft_train.jsonl`，`--num_train_epochs 1`、`--gradient_accumulation_steps 8`、`--output_dir /data/runs/sft-swift`）；GRPO 把 `--max_steps 2` 改成 200–300、`--gradient_accumulation_steps 4`。训练任务的启动命令（Git 来源要先接线）：

```bash
ln -sfn /data/assets/course-dataset /code/workspace/Dataset; ln -sfn /data/assets/models /code/workspace/models; bash /code/workspace/scripts/run_swift_grpo.sh
```

最后跑评测（推理镜像）：`baseline.py --shard i/4` → `merge_predictions.py` 合并重算；提交用 `export_submission.py`。

## 7. 已知坑

- **transformers 版本**：训练镜像别抄推理镜像的 pin；adapter 是标准 PEFT 格式，合并后在 5.20.2 推理栈里加载，冒烟第 4 条就是验证这一点。
- **音频**：VA 用独立 wav，靠 `<audio>` 占位符，不靠 `USE_AUDIO_IN_VIDEO`（镜像里已设 `false`）。VT 行没有 audio 字段。
- **视频采样**：ms-swift 用 decord 自己采帧，和评测的 8 帧/448 不完全一致；训练/评测预处理差异可接受，但报告里要写明。
- **vLLM**：先不开，HF rollout 足够跑 3B；要加速再加 `--use_vllm true --vllm_mode colocate`，并重新冒烟。
- **云盘**：audio 目录约 1–2 GB；只存 adapter，合并权重写 `/data/tmp` 评完删。

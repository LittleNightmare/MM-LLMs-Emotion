# MM-LLMs-Emotion 项目指引

《人工智能原理与应用》课程“大模型后训练”作业：在统一多模态情感数据上，比较原始模型、监督微调（SFT）、强化学习（RL）的情感识别、模态冲突识别和过度判断率；证据不足时表达不确定、提出澄清问题，并通过交互 Demo 展示用户补充回答后的判断更新。

## 要求与当前进度

- 作业原文：`F:\BUAA\课堂\人工智能原理与应用\大模型后训练.pdf`；仓库任务清单与提交材料要求见 `README.md`。
- 已完成：数据下载与核对、数据格式整理、媒体预处理、五组零样本基线和实验归档；结果详见 `docs/基线实验.md`（2026-10-21）。
- 当前仓库尚无 SFT、RL 或 Demo 实现；README 中 Omni-3B LoRA 是待验证方案。
- 数据已到位：`Dataset/` 于 2026-10-27 从 Kaggle 竞赛页下载（2974 视频 + 3 个 JSONL + 提交样例，1.57 GB）；四项缺口（不确定性标签、澄清问题质量标注、多轮判断更新案例、过度判断率定义与评价）仍待完成，详见 `docs/数据状态与缺口.md`。
- Kaggle 竞赛口径（总分、每日提交次数、Public/Private 划分、数据使用限制）见 `docs/kaggle-竞赛与提交.md`；竞赛不评追问质量，也不含过度判断率。
- 先核对补标规则、审计训练/验证/测试视频与衍生样本重合；保留官方划分，记录训练排除清单，补标训练与评价分离，再做小规模训练与保存/重载检查。
- PDF 要求自设计 RL 奖励函数，并比较三阶段的情感 Macro-F1、冲突准确率、过度判断率和典型案例。

## 环境与安装

- 实验环境：Python 3.12、PyTorch `2.13.0+cu132`、torchvision `0.28.0+cu132`、Transformers `5.20.2`，由仓库根目录 `pyproject.toml` 固定、`uv.lock` 锁定；旧版 `requirements-baseline.txt` 保留作对照。平台镜像另走 Python 3.11 + torch 2.7.1（`docker/`），依赖版本已对齐 3.11（`av==18.1.0`、`librosa==0.10.2`）。
- 用 uv 建环境：`uv sync` 在仓库根目录生成 `.venv`，同时装上 kaggle CLI 与 jupyterlab；国内默认走清华 PyPI 与 SJTU pytorch-wheels 镜像，换到官方源后需重新 `uv lock`。
- 本机 GPU 为 RTX 4070 Laptop（8 GB），CUDA 版 torch 可正常 `torch.cuda.is_available()`；长训练与提交结果以 Intern Discovery 平台为准（自定义镜像，见 `docker/`）。
- 所有命令从仓库根目录执行；本地跑脚本用 `uv run python ...`，不要往 Anaconda base 装包。
- 推理使用 CUDA 设备 0 和 BF16，直接调用 CUDA API，没有 CPU 回退；历史测试使用单张 48GB RTX 4090。
- `Dataset/` 已在本地就位；模型权重仍需另行准备，历史配置中的 `/workspace/...`、`/model/...` 是实验服务器路径。

```bash
uv sync
uv run python scripts/test_baseline_metrics.py
```

Kaggle 与 HF 凭据优先读仓库根目录 `.env`（模板 `.env.example`，`.env` 已忽略），没有时退回 `~/.kaggle/access_token` 与 HF 登录态；下载与提交命令见 README 第 3 节。

## 运行与检查

指标回归测试入口使用标准库 unittest；覆盖非法输出计错、缺失类别和 JSON 解析：

```bash
python scripts/test_baseline_metrics.py
```

测试会导入 `baseline.py` 顶层的 PyTorch、Transformers、音视频依赖，即使不加载模型也需要这些包。仓库没有单独的 build、lint 或格式化命令。

现有文档中的 VT 推理命令（先准备对应模型目录及验证数据）：

```bash
python scripts/baseline.py --model qwen35-2b --protocol VT --model-path models/qwen35-2b --output outputs/qwen35-2b
```

- `--smoke` 按协议、一致/冲突、真实/生成来源各取少量样本；使用独立输出目录，不能把 smoke 分数当完整验证结果。
- `scripts/audit_baselines.py` 校验完整覆盖、输入模态、共享设置和媒体/脚本/数据哈希；需要原始数据、各 run 目录及缓存，输出父目录需已存在。
- `scripts/checkpoint_evidence.py` 保存权重哈希和来源；只有所有权重哈希与官方一致时才记录已验证的 HF commit。
- `scripts/compose_protocol_baseline.py` 只按 VT/VA 组合已有预测，不重新推理；目标输出目录必须尚不存在。

## 数据与输出约定

- `Dataset/train.jsonl`、`val.jsonl`、`test_release.jsonl`：分别为 3354、666、734 条；数据已下载到本地 `Dataset/`，由 `.gitignore` 排除，不随仓库提交。
- 训练/验证每行包含 `sample_id`、`protocol`、`video`、`sample_type`、`views`；VT 另读 `text_content`。`video` 相对标注文件所在目录解析。
- 情感监督取 `views.M12.label`，允许值为 `negative / positive / neutral`；冲突标签取 `sample_type == "Conflict"`。
- VT 输入无声音画面和给定文字；VA 输入画面及完整音轨，不额外提供文字稿。标签仅用于生成后评分。
- 视频默认全片均匀采样 8 帧、最长边 448；音频为 16kHz 单声道；BF16、贪心解码、192 新 token、种子 42。
- 输出 JSON 字段固定为 `pred_emotion`、`is_conflict`、`is_uncertain`、`clarifying_question`；两个布尔含义字段使用字符串 `yes/no`，无需追问时填字符串 `none`。
- 每次推理输出 `config.json`、`baseline_source.py`、`model_revision.json`、`predictions.jsonl` 和持续更新的 `metrics.json`。
- 测试集只有 `sample_id/video/text`：`baseline.py` 自动推断协议（ID 第二段，生成片段看 `video_audio` 目录），无标签时只输出描述性统计；`scripts/export_submission.py` 生成并校验提交 CSV（四列 `sample_id,pred_emotion,is_uncertain,clarifying_question`）。

## 代码与实验记录

- `scripts/baseline.py` 集中实现预处理、JSON 解析、指标及模型推理；同目录脚本通过 `from baseline import ...` 导入，测试同样直接以脚本运行。
- Python 使用四空格缩进、snake_case；文档主要为中文 Markdown；JSON 使用 UTF-8，实验记录保留中文。
- `.gitattributes` 指定 Python、JSON、Markdown、TXT 使用 LF，`.gz` 为二进制。
- `.gitignore` 排除 `Dataset/`、`models/`、`outputs/`、`evidence/`、`logs/`、`.venv/` 和 Python 缓存。
- 已发布基线保存在 `results/small-baselines-20261021/`；原始预测、日志及当时脚本位于 `experiment-evidence.tar.gz`，附 SHA-256 文件。新实验保留独立记录。
- 归档内包含当前工作树没有的旧报告/下载/打包脚本；历史文档提到这些文件不表示当前可直接调用。

## 易踩的坑与评价口径

- Qwen3.5 无音频编码器，必须显式 `--protocol VT`；Omni/Gemma 的全量结果与 Qwen3.5 的 VT 结果需按相同子集比较。
- 基线指尚未经过本课程任务训练的官方指令/后训练权重，不代表未经指令训练的预训练 Base。
- 断点续跑要求配置完全一致，包括脚本/数据哈希和版本；修改参数、脚本或运行环境后换新输出目录。已有预测按 sample_id 跳过，失败样本也不会自动重试。
- 并发预处理使用不同 `--media-cache` 目录，避免同时写入同名媒体文件；同一输出目录也不支持多进程追加。
- 连续三条推理异常会停止；`metrics.json` 为已完成样本的累计指标，不能仅因文件存在就认为完整评测结束。
- 验证集没有 neutral：正文 Macro-F1 使用 `emotion_macro_f1_present`（实际出现的两类）；`emotion_macro_f1_fixed3` 另含零支持的 neutral，不能混用。
- 无法解析或非法标签仍计入分母并按错误计；JSON 合格率单独统计，不能丢弃格式错误后重新报告准确率。
- 验证集 666 条中只有 74 条冲突，需同时看冲突 Precision/Recall/F1；高冲突准确率不代表模型能检出冲突。
- 冲突不等于必须追问；问题允许多种有效措辞，用户回答后允许保持原判或仍不确定。`uncertain_rate` 不能当不确定性准确率或过度判断率；候选指标定义不能当官方规范。
- VT 用 Qwen3.5-2B、VA 用 Omni-3B 的组合是观察验证结果后的离线组合，没有独立测试或组合部署显存/延迟实测。
- 推理显存记录为 PyTorch 已分配显存峰值，不能用于断言训练显存需求；ModelScope 的可变 `master` 路径也不能替代权重哈希。

# 多模态情感识别与大模型后训练

## 1. 课程信息与题目要求

本项目是《人工智能原理与应用》课程大作业，选择“大模型后训练”方向。

输入视频画面及文字或音频，模型需要判断人物情感、识别模态冲突；当证据不足时表达不确定并提出澄清问题，再根据用户补充的信息更新判断。

需要完成：

- 比较原始模型、监督微调（SFT）和强化学习（RL）三个阶段的表现。
- 自行设计 RL 奖励函数，兼顾情感判断、冲突识别和合理追问。
- 评测情感识别 Macro-F1、模态冲突识别准确率和过度判断率。
- 制作交互 Demo，展示模型判断、追问以及补充回答后的结果变化。

按课程材料，10 人一组，第 8～10 周汇报；提交包含 Demo 的 PPT（汇报不超过 15 分钟）、不少于 4000 字的 Word 技术报告，以及项目代码。报告需包含 GitHub 仓库链接。

| 资料 | 链接 |
| --- | --- |
| 课程数据集 | [人工智能原理与应用-数据集](https://bhpan.buaa.edu.cn/link/AAA63CDEC9ED3B48B2A7C64CB5911C04AD) |
| Kaggle 竞赛 | [大模型后训练](https://www.kaggle.com/competitions/video-emotion-uncertainty-2026)（邀请链接 `https://www.kaggle.com/t/6807427c8d58473e914ea43f81b90bd1`） |
| 作业提交 | [大作业](https://bhpan.buaa.edu.cn/link/AACBE9D036E1CF421FAB26A6891595076B) |

### Kaggle 竞赛

总分 `0.5 × 情感 Macro-F1 + 0.3 × 冲突识别准确率 + 0.2 × 一致样本校准率`。冲突识别准确率统计真冲突样本里 `is_uncertain=yes` 的比例，一致样本校准率统计真一致样本里 `is_uncertain=no` 的比例；两项只看不确定性字段，不看情感判断是否判对，`clarifying_question` 不参与评分。比赛期间排行榜按 Public 70% 计分，结束后按 Private 30% 排名；排行榜分数仅供参考，最终成绩以实验报告和 Demo 为准。

提交为 `sample_id,pred_emotion,is_uncertain,clarifying_question` 四列，覆盖全部 734 条样本；`pred_emotion` 取 `positive/negative/neutral`，`is_uncertain` 取 `yes/no`，无需追问时 `clarifying_question` 填 `none`，留空会被拒绝；每天最多提交 10 次。

数据仅限本课程作业使用，禁止外传、公开或商用；禁止以任何方式获取测试集标签，包括人工看视频标注后针对性提交。评分口径、字段对照与提交流程见 [Kaggle 竞赛与提交](docs/kaggle-竞赛与提交.md)。

## 2. 数据集格式

数据放在 `Dataset/`，标注文件为 JSONL，每行一条样本。同一视频可以分别对应两种任务：

- **VT（Video + Text）**：输入画面和给定文字，不输入音轨。
- **VA（Video + Audio）**：输入画面和音频，不额外输入文字稿。

| 文件 | 用途 | 样本数 | VT | VA |
| --- | --- | ---: | ---: | ---: |
| `train.jsonl` | 训练 | 3354 | 1780 | 1574 |
| `val.jsonl` | 验证 | 666 | 345 | 321 |
| `test_release.jsonl` | 测试，无公开标签 | 734 | 385 | 349 |
| `sample_submission.csv` | 提交格式示例 | 734 | — | — |

`videos/` 共 2974 个 mp4、143 个目录，其中 `videos/gen/` 是合成补充片段；整个数据集 2978 个文件、1.57 GB，采用 CC BY-NC-SA 4.0，仅限课程使用。

训练、验证样本的字段如下（示例省略了部分元数据字段）：

```json
{
  "sample_id": "ch_sims_v2:VT:video_0001:0001",
  "protocol": "VT",
  "video": "videos/video_0001/0001.mp4",
  "text_content": "我不想嫁给李茶",
  "sample_type": "Aligned",
  "dominant_modality": "balanced",
  "source_dataset": "ch_sims_v2",
  "source_is_generated": false,
  "views": {
    "M1": {"label": "negative", "is_clear": true},
    "M2": {"label": "negative", "is_clear": true},
    "M12": {"label": "negative", "is_clear": true}
  }
}
```

`video` 是相对于 `Dataset/` 的路径。M1 为画面标签，M2 为文字或音频标签，M12 为综合情感标签。情感分为 `negative / positive / neutral`；`sample_type` 分为一致（Aligned）和冲突（Conflict）。

另有 `quality_flags`、`use_in_main`、`label_basis`、`split_group_id`、`annotator_agreement`、`annotation_count` 等字段；`media_paths` 是数据构建机器的绝对路径，在本机不可用，取视频一律用 `video`。每个视角还带 `specific_affect`（本批为空）和 `is_clear`，但本批数据 `M12.is_clear` 全为 true，不能用它当不确定性标签。训练集 3354 条中冲突 411 条（约 12%），中性样本仅 7 条。

测试集只有 `sample_id`、`video`、`text`（训练/验证里同一内容叫 `text_content`），没有 `protocol` 字段：`ch_sims_v2` 的 ID 第二段即协议，生成片段按视频目录区分，`silent_video_text` 记 VT、`video_audio` 记 VA。提交文件包含 `sample_id,pred_emotion,is_uncertain,clarifying_question` 四列，本地评测另外记录 `is_conflict`。

**数据状态：已有三份划分和情感/冲突监督，尚未满足完整作业的数据需求。**`Dataset/` 已于 2026-10-27 从 Kaggle 竞赛页下载完整数据（2974 个视频 + 3 个 JSONL + 提交样例），样本数、VT/VA 与冲突条数均已按本地文件复核。跨划分的视频及衍生样本重合尚待审计，不能把“已划分”等同于“无泄漏”。

| 尚缺内容 | 当前状态 | 需要补齐 |
| --- | --- | --- |
| 不确定性判断 | 无独立标签 | 证据充分性、是否需补充信息及人工判定依据 |
| 澄清问题 | 无参考问题及质量规范 | 缺失信息、可接受问题和有效性评价；不只匹配固定措辞 |
| 追问后更新 | 无多轮监督及独立评价案例 | 问题、用户回答、更新结果；允许保持原判或仍不确定 |
| 过度判断率 | 定义、评价依据和实现均待完成 | 固定分子/分母、人工评价集及计算；不能以不确定输出比例替代 |

验证集没有中性样本，当前 Macro-F1 按正向、负向两类计算；冲突只有 74 条，需同时看冲突召回率与 F1。四项缺口的完成条件、划分检查和补标约束见[数据状态与缺口](docs/数据状态与缺口.md)。

## 3. 环境与依赖

依赖写在 [pyproject.toml](pyproject.toml)，`uv sync` 在仓库根目录建 `.venv`（Python 3.12、torch 2.13.0+cu132、torchvision 0.28.0+cu132，另含 kaggle CLI 和 jupyterlab），`uv.lock` 保证多人装到同一套版本。默认走清华 PyPI 和 SJTU 的 pytorch-wheels 镜像，海外协作时把 `pyproject.toml` 里两个 index 换回官方地址再重新 `uv lock`。

```bash
uv sync
uv run python scripts/test_baseline_metrics.py
```

凭据放仓库根目录的 `.env`（从 `.env.example` 复制一份改），`.env` 已被 `.gitignore` 排除，不会进仓库；不填也能跑，脚本会退回 `~/.kaggle/access_token` 和 Hugging Face 的登录态。支持 `KAGGLE_API_TOKEN`、`HF_TOKEN`、`HF_ENDPOINT` 三个键。数据和权重都走同一个脚本：

```bash
uv run python scripts/fetch_assets.py dataset                     # 下载并校验数据（734/666/3354、2974 个视频）
uv run python scripts/fetch_assets.py models qwen-omni-3b --dry-run  # 先看体积，不下载
uv run python scripts/fetch_assets.py models qwen-omni-3b          # 拉基座权重到 models/
uv run python scripts/fetch_assets.py all qwen-omni-3b             # 数据 + 权重
```

`models --list` 列出候选模型。按当前磁盘余量，只建议先下 `qwen-omni-3b`（约 12 GB）；`qwen-omni-7b`、`gemma-e2b`、`qwen35-2b` 等放到服务器或清理空间后再拉。测试集推理与提交：

```bash
# 测试集没有标签，protocol 由 sample_id 和视频目录推断
uv run python scripts/baseline.py --model qwen-omni-3b --dataset Dataset/test_release.jsonl --output outputs/test
# 生成并校验提交 CSV，任何格式错误都会拦住
uv run python scripts/export_submission.py --predictions outputs/test --output submission.csv
uv run kaggle competitions submit -c video-emotion-uncertainty-2026 -f submission.csv -m "说明"
```

本地 `.venv` 用于冒烟测试和脚本调试；需要 GPU 的长训练和提交结果以 Intern Discovery 的平台为准，那边用自定义镜像（Python 3.11 + torch 2.7.1，见 [docker/](docker/)），不再装 CUDA 版 torch。

## 4. Plan

### 数据与基线

- [x] 整理数据划分、字段和 VT / VA 输入方式。
- [x] 从 Kaggle 下载完整数据集到 `Dataset/`，按本地文件复核样本数、VT/VA 与视频可读性。
- [x] 用 uv 固定依赖并建好本地环境。
- [x] 跑通视频采样、音频处理和模型推理。
- [x] 完成 Gemma 4 E2B、Qwen2.5-Omni-3B / 7B、Qwen3.5-0.8B / 2B 的基线测试，其中 Qwen3.5 仅测 VT。
- [x] 保存原始回答、配置、指标和日志，整理[基线实验结果](docs/基线实验.md)。

### 训练准备与 SFT

- [ ] 确定训练基座与单卡配置，优先验证 Omni-3B 的 LoRA 训练。
- [x] 核对课程/Kaggle 补标与外部数据规则，见 [Kaggle 竞赛与提交](docs/kaggle-竞赛与提交.md)。
- [ ] 审计训练、验证、测试间视频及衍生样本重合，建立训练排除清单。
- [ ] 补齐独立的不确定性标签与证据充分性判定规范。
- [ ] 补齐澄清问题示例、信息需求与问题质量评价规范。
- [ ] 补齐用户回答后的判断更新案例，覆盖改判、保持原判和仍不确定。
- [ ] 定义过度判断率，建立独立人工评价集并实现计算；评价标签不用于训练或奖励调参。
- [ ] 构建 SFT 数据，覆盖情感分类、模态冲突和澄清回答。
- [ ] 小规模试训，检查显存、模型保存和重新加载。
- [ ] 完成 SFT，在相同验证集和推理设置下与基线比较。

### RL 与评测

- [ ] 设计并检查奖励函数，覆盖情感正确性、冲突识别、输出格式和合理追问。
- [ ] 从 SFT 模型继续 RL 训练，保存训练配置和模型。
- [ ] 统一评测原始模型、SFT、RL，分别报告 VT、VA 和完整任务结果。
- [ ] 分析错误案例、冲突样本及追问表现，比较不同奖励设置。
- [x] 核对 Kaggle 评分与提交规则，记录口径与数据使用限制。
- [x] 适配测试集格式（`test_release.jsonl` 只有 `sample_id/video/text`），写生成与校验脚本。
- [ ] 在测试集上跑推理，生成 `submission.csv` 并上传。

### Demo 与提交

- [ ] 完成交互 Demo，支持输入视频、模型追问和用户补充回答。
- [ ] 整理训练与评测命令、模型和实验结果。
- [ ] 完成技术报告、汇报 PPT 和演示，提交作业。

各项实验的设置、训练参数和汇总结果见[实验日志](docs/实验日志.md)。基线详情见[基线实验](docs/基线实验.md)，原始基线记录见 [results](results/small-baselines-20261021/)，推理入口为 [scripts/baseline.py](scripts/baseline.py)。

# 课程参考文献

本目录保存课程 PDF 指定的三篇论文、从 Zotero 复制的原始 PDF、MinerU 全页解析结果，以及与作业要求的对照说明。Zotero 来源集合：`北航/课堂/人工智能`（集合 key：`N6XHERLE`）。

先读 [作业要求与参考论文对照](作业要求与参考论文对照.md)。

| 论文 | 本地 PDF | MinerU 全文 | PDF 页数 | 与作业的关系 |
| --- | --- | --- | ---: | --- |
| Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning | [PDF](emotion-llama/emotion-llama.pdf) | [Markdown](parsed/emotion-llama.md) | 49 | 情感指令数据构建、多任务 SFT、模态线索分析 |
| R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning | [PDF](r1-omni/r1-omni.pdf) | [Markdown](parsed/r1-omni.md) | 9 | 冷启动 SFT、GRPO、情感正确性与格式奖励 |
| MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition | [PDF](mer2024/mer2024.pdf) | [Markdown](parsed/mer2024.md) | 8 | 数据、竞赛任务、噪声鲁棒性、开放词汇评价；MERTools 是配套工具箱 |

## 文件结构

- `emotion-llama/`、`r1-omni/`、`mer2024/`：原始 PDF 和按 PDF 页码分隔的 `text-layer.md`，方便核对原文。
- `parsed/`：MinerU 生成的 Markdown、JSON 结构化内容和 `images/`；JSON 的 `page_idx` 从 0 开始。
- `sources.json`：Zotero 条目/附件 key、PDF 页数与 SHA-256。
- `verification.json`：解析命令、工具版本、逐篇页覆盖与图像链接检查结果。

## 解析方法与核验

使用 `mineru-open-api v0.5.9` 的 precision `extract` 模式，`vlm` 模型，不传 `--pages`，处理全部页（包括参考文献、附录及 Emotion-LLaMA 的 NeurIPS checklist）：

```bash
mineru-open-api extract reference/emotion-llama/emotion-llama.pdf reference/r1-omni/r1-omni.pdf reference/mer2024/mer2024.pdf --model vlm --language en --format md,json --output reference/parsed --timeout 1800
```

三篇全部成功。已核对：复制 PDF 与 Zotero 原件哈希一致；JSON 覆盖全部 PDF 页；Markdown 的本地图像引用全部存在；解析末尾与各 PDF 最后一页的内容相符。页覆盖检查不代表逐字校订；VLM 提取可能出错，具体论断与表格以原 PDF 为准。

论文自身也有口径不一致：MER 2024 的 MER-OV 数量表中为 322、正文为 332；R1-Omni 正文部分 UAR/WAR 标签与表 1 对调。详见对照说明，不直接修改原始解析文件掩盖这些差异。

这里只存论文与解析材料，没有下载论文数据集、模型权重或训练作者模型。目录用于本地学习，本次未提交或推送；原始论文、插图和相关数据的再分发按各自许可处理。

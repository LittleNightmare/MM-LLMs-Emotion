# 小模型实验文件

2026-10-21。本轮新增 1356 条推理预测：Qwen3.5-0.8B 和 2B 各 345 条 VT，Qwen2.5-Omni-3B 共 666 条 VT＋VA。另有 666 条按协议组合的预测，复用已有回答，没有新增推理。

数据、模型属性、实验方式和结果见 [基线实验](../../docs/基线实验.md)。

## 1. 配置和指标

本目录的 `*-config.json`、`*-metrics.json`、`*-model_revision.json` 保存每个实验的配置、完整指标及实际加载参数。`confusion_counts.json` 保存分类混淆统计和格式问题诊断。旧的 Gemma E2B、Omni-7B 也一并保留。

## 2. 原始证据

`experiment-evidence.tar.gz` 包含逐条原始回答、评分结果、运行日志、实际执行的脚本副本、模型配置、权重哈希及官方版本核对结果。

`experiment-evidence.tar.gz.sha256` 是压缩包的 SHA256；`evidence-files.json` 校验包内历史快照，不用于校验后来整理的仓库文档。包中沿用仓库相对路径，原始回答位于 `outputs/`，日志位于 `logs/`。解压到独立目录查看：

```bash
mkdir -p evidence
tar -xzf results/small-baselines-20261021/experiment-evidence.tar.gz -C evidence
```

本地仓库和服务器 `/workspace/MM-LLMs-Emotion` 各保留一份。模型权重和课程视频没有放进压缩包。

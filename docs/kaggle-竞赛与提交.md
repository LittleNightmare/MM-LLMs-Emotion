# Kaggle 竞赛与提交

记录时间：2026-10-27。依据竞赛页 Overview / Data / Rules 页面的原文，以及本机下载后的数据核对。

## 基本信息

| 项 | 内容 |
| --- | --- |
| 竞赛 | 大模型后训练（视频情感大模型），Community Prediction Competition，Private |
| 竞赛页 | <https://www.kaggle.com/competitions/video-emotion-uncertainty-2026> |
| 邀请链接 | `https://www.kaggle.com/t/6807427c8d58473e914ea43f81b90bd1`（课程提供，登录后加入并接受规则） |
| 主办 | Sunny_6667 |
| 时间 | 2026-10-13 开始，2026-10-30 结束 |
| 数据 | 2978 个文件、1.57 GB，CC BY-NC-SA 4.0 |
| 榜单 | Public 70% / Private 30%，结束后按 Private 排名 |

## 评分口径

总分（满分 1.0）：

```
0.5 × Macro-F1 + 0.3 × 冲突识别准确率 + 0.2 × 一致样本校准率
```

| 指标 | 权重 | 定义 |
| --- | ---: | --- |
| Macro-F1 | 0.5 | `pred_emotion` 与真值的宏平均 F1 |
| 冲突识别准确率 | 0.3 | 真冲突样本（`sample_type=Conflict`）中 `is_uncertain=yes` 的比例 |
| 一致样本校准率 | 0.2 | 真一致样本（`sample_type=Aligned`）中 `is_uncertain=no` 的比例 |

三点容易误解的地方：

- 两项不确定性指标只看 `is_uncertain`，不检查情感判断是否正确。把冲突样本的情感判错但标了 `yes`，这一项仍然满分。
- 一致样本校准率是为了防止“全部填 `yes`”刷分；全部填 `no` 同样会在冲突项拿零分。
- 全部填 `neutral/no/none` 的样例提交得 0.200，即只拿到了校准率那 0.2。

竞赛评分不包含课程作业要求的“过度判断率”，也不评价 `clarifying_question` 的实际质量。报告里的过度判断率和追问质量需要另行定义与人工评价，不能拿这套榜单分数代替。

## 提交规则

- 文件格式见 `Dataset/sample_submission.csv`，四列：`sample_id,pred_emotion,is_uncertain,clarifying_question`。
- 必须覆盖测试集全部 734 条样本。
- `pred_emotion` 只能取 `positive` / `negative` / `neutral`；`is_uncertain` 只能取 `yes` / `no`。
- `clarifying_question` 在不需要追问时填 `none`；任何位置留空都会被拒绝。
- 每天最多 10 次提交。

## 竞赛规则与数据使用

- 数据集仅限本课程作业使用，禁止外传、公开传播、上传到其他平台或用于商业用途。
- 禁止以任何方式获取测试集标签，包括人工看完视频后针对性提交。
- 同时受 Kaggle Competition Foundational Rules 约束，两者冲突时以后者为准；Foundational Rules 还规定：一个账号只能提交自己的结果，团队内可以共享代码，跨团队私下共享代码禁止，公开分享的代码需放在该竞赛的讨论区或 Notebook。

## 数据与提交字段对照

| 提交列 | 来源 | 说明 |
| --- | --- | --- |
| `sample_id` | `test_release.jsonl` | 734 条，顺序不限 |
| `pred_emotion` | 模型判断 | 三分类，须与 `views.M12.label` 取值范围一致 |
| `is_uncertain` | 模型不确定性输出 | 竞赛的真值是 `sample_type`，但提交时只填 yes/no |
| `clarifying_question` | 模型追问 | 无追问填 `none`，不参与评分 |

训练/验证评价用的 `is_conflict` 不在提交列里，只用于本地指标。

## 本地运行

依赖由仓库根目录的 `pyproject.toml` 管理，`uv sync` 建 `.venv`：

```bash
uv sync
uv run kaggle competitions download -c video-emotion-uncertainty-2026 -p Dataset
uv run kaggle competitions submit -c video-emotion-uncertainty-2026 -f outputs/submission.csv -m "说明"
```

凭据优先读仓库根目录的 `.env`（从 `.env.example` 复制，`KAGGLE_API_TOKEN` 填 `KGAT_` 开头的新式 token），没有 `.env` 时退回 `~/.kaggle/access_token`；`.env` 已在 `.gitignore` 里。课程给的数据压缩包已经解开到 `Dataset/`，日常不需要重复下载。

## 当前状态

- 账号已加入竞赛并接受规则（2026-10-27）。
- 数据已下载到 `Dataset/`：训练 3354 / 验证 666 / 测试 734，2974 个视频全部可读，`video` 相对路径无缺失。
- 尚未生成提交文件；测试集只有 `sample_id/video/text`，现有推理入口依赖训练/验证的标签结构，需要先适配。
- 排行榜分数只作参考，课程明确最终成绩以实验报告和 Demo 为准。

# Intern Discovery 平台使用

记录时间：2026-10-27。来源是平台帮助中心的[科学计算概述](https://discovery.intern-ai.org.cn/docs/workbench/overview)、[我的开发机](https://discovery.intern-ai.org.cn/docs/workbench/machine)、[为开发机准备数据](https://discovery.intern-ai.org.cn/docs/workbench/data-prepare)、[我的云盘](https://discovery.intern-ai.org.cn/docs/workbench/cloud)、[训练任务](https://discovery.intern-ai.org.cn/docs/workbench/training)、[镜像](https://discovery.intern-ai.org.cn/docs/workbench/images)。没有在平台上实测，标“待验证”的都要开机确认。

## 一、它提供什么

科学计算 = 开发机 + 云盘 + 训练任务 + 镜像。开发机是带 GPU 的云端 IDE（默认 VSCode/code-server，可切 JupyterLab），云盘挂在 `/data` 做持久化，训练任务是托管跑批。

| 项 | 规格 / 限制 |
| --- | --- |
| 资源档位 | NvidiaA100-1-80G、NvidiaA100-8-80G、Ascend910B-1-64G、Ascend910B-8-64G、CPU-4C-16G（4 核 / 16G 内存） |
| 开发机数量 | 单账户最多 10 台 |
| 单次运行时长 | 最长 7×24 小时；停止后重新启动要重设时长 |
| 云盘 | 默认 30 GB，可申请扩容；挂载到开发机的 `/data` |
| 数据集 | 挂载到只读的 `/dataset`，最多 5 个 |
| 自建数据集上传 | 文件数 < 10、单文件 ≤ 50 MB，超限用数据集 CLI |
| 镜像 | 最多 5 个；快捷安装 apt/pip 或写 Dockerfile；开发机镜像与训练任务镜像分开 |
| 计费 | 算力点，按资源 × 时长；停机不扣 |

## 二、存储怎么摆

只有 `/data` 持久化，其余目录在开发机停止或释放后重置。代码、数据、模型、venv 全部放 `/data`，目录按“大件只放一次”来分：

```
/data/
├── assets/
│   ├── Dataset/               # 竞赛数据，1.57 GB，只下一次
│   └── models/qwen-omni-3b/   # 基座权重，约 12 GB，只下一次
├── MM-LLMs-Emotion/           # 代码，几百 KB，可反复覆盖
│   ├── Dataset -> /data/assets/Dataset
│   └── models  -> /data/assets/models
├── env/mm-llms-emotion/       # uv 的 venv
└── runs/                      # 输出、checkpoint
```

软链接让 `baseline.py` 这类写死 `Dataset/` 的脚本不用改。`fetch_assets.py` 也支持直接指定 `--dataset-dir /data/assets/Dataset`、`--models-dir /data/assets/models`。

配额估算：1.57 + 12 + venv 3.5 + 代码 ≈ 17.5 GB，30 GB 够用；再加 7B（16 GB）就要扩容，或者把竞赛数据传成平台数据集挂到 `/dataset`（只读、不占云盘）。

## 三、环境：不急着做镜像

官方镜像预装 PyTorch 2.4.0 / Python 3.11，和仓库钉的 `torch 2.13.0+cu132` / Python 3.12 不一致。建议先用 uv 在云盘里建环境：

```bash
export UV_PROJECT_ENVIRONMENT=/data/env/mm-llms-emotion
uv sync
```

venv 落在 `/data`，后续开发机和训练任务都能复用同一份，只装一次。做镜像要选资源配置、消耗算力点，还要占 5 个镜像名额，在额度紧张时不划算。

需要自建镜像的情况：训练任务每次都要重装依赖、或镜像自带 torch 与机器驱动不匹配导致跑不起来。那时用 Dockerfile 基于官方镜像 pip 装 `pyproject.toml` 里的依赖即可，构建一次反复用。

## 四、代码怎么来回

仓库用 [scripts/sync_code.py](../scripts/sync_code.py) 打包，默认排除 `Dataset/`、`models/`、`reference/`、`.venv/`、`.git`，产物约 100 KB：

```bash
uv run python scripts/sync_code.py pack          # dist/mm-code-<时间>.tgz
uv run python scripts/sync_code.py upload        # 传到云盘 /MM-LLMs-Emotion/
```

机器上解包并接线：

```bash
mkdir -p /data/MM-LLMs-Emotion/repo
tar -xzf /data/MM-LLMs-Emotion/mm-code-*.tgz -C /data/MM-LLMs-Emotion/repo
ln -s /data/assets/Dataset /data/MM-LLMs-Emotion/repo/Dataset
ln -s /data/assets/models  /data/MM-LLMs-Emotion/repo/models
```

需要保留 git 历史时用 `pack --with-git`，或者走 `git bundle`：本地 `git bundle create mm.bundle --all`，机器上 `git clone mm.bundle`，回传再 `git fetch`。GitHub 能不能直连还没验证，先按不通准备。

`discovery-ctl` 的安装方式文档里没写；开发机大概率自带（进机器先 `which discovery-ctl`），本地要用得找平台给的下载入口。没有 CLI 时，网页云盘拖拽上传同样可以，只是慢一些。

## 五、第一次进机先跑这几条

```bash
nvidia-smi                                  # 驱动版本，决定 cu132 轮子能不能用
python -V && python -c "import torch; print(torch.__version__)"
which git uv discovery-ctl
df -h /data
curl -I https://www.kaggle.com && curl -I https://hf-mirror.com && curl -I https://pypi.tuna.tsinghua.edu.cn
```

外网出口是最关键的未知项：通不通决定数据和模型是在机器里直接下，还是本地传上去。

## 六、训练任务怎么配

- 代码来源选“我的云盘”，路径填 `/MM-LLMs-Emotion/repo`，启动命令写 `bash /data/MM-LLMs-Emotion/repo/scripts/xxx.sh`；选 Git 来源时工作目录是临时的 `/code/workspace`。
- 结果和 checkpoint 必须写 `/data`，任务停止后不能续跑，只能“重新训练”，断点续训靠 `/data` 里的 checkpoint。
- TensorBoard 日志写到 `TENSORBOARD_LOGDIR` 指定的目录，任务详情页才有曲线。
- 单次最长 7×24；额度有限时优先长任务，别拿 GPU 机器做数据处理。

## 七、待验证

- 开发机与训练任务的外网出口（Kaggle / Hugging Face / PyPI）。
- A100 机器驱动能否直接跑 `cu132` 轮子；不行就退 cu126/cu128，或用镜像自带 torch 2.4 跑推理。
- `discovery-ctl` 在本地 Windows 上的安装包与登录方式。
- 云盘实际可用容量与扩容审批速度。

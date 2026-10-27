# 平台镜像

给 Intern Discovery 的开发机和训练任务用的环境镜像。目标是**开机就有依赖**，不用每次 `uv sync` 或 `pip install`。

## 两条路

| 方式 | 适合 | 说明 |
| --- | --- | --- |
| 快捷安装 | 只缺 Python 包和系统包 | 在镜像页填 apt / pip 清单，见 `apt-packages.txt`、`pip-packages.txt` |
| Dockerfile | 要装系统工具、设环境变量、跑自检 | 用同目录 `Dockerfile`，平台会自动填 `FROM` |

## 平台上的限制

- **使用场景分开发机和训练任务**，镜像建好后只会出现在选定场景里。两边都要用就建两个，占 2 个名额。
- 最多 5 个镜像，构建要选资源配置、消耗算力点。
- 基础镜像跟资源配置绑定：选 Nvidia GPU 才会列出 CUDA / PyTorch 环境，选 CPU 只有 CPU 环境。

## 基础镜像怎么选

优先选平台提供的 PyTorch 基础镜像，按 A100 那台机器的驱动来挑 CUDA 版本。镜像自带的 `torch` / `torchvision` 直接用，不在这里另装——`cu132` 这类新轮子要求驱动足够新，装之前先看 `nvidia-smi` 的驱动版本。

如果后面确认驱动支持 CUDA 13，可以在 Dockerfile 里再加一步升级：

```dockerfile
RUN pip install --no-cache-dir torch==2.13.0+cu132 torchvision==0.28.0+cu132 \
      --index-url https://download.pytorch.org/whl/cu132
```

## 依赖清单从哪来

`pip-packages.txt` 和 `Dockerfile` 里的版本跟仓库 `pyproject.toml` 对齐，改依赖时两边一起改。`torch` / `torchvision` 不在清单里，由基础镜像提供。

## 建好之后怎么用

镜像构建成功后，新建开发机或训练任务时在「镜像」里选你的镜像即可，依赖已经就位，不用再跑 `uv sync`：

```bash
# 开发机里直接跑
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/fetch_assets.py models qwen-omni-3b --dry-run
```

`machine_bootstrap.sh` 加 `--no-uv` 就是这个模式：跳过 uv，用系统 Python 只做目录、软链和体检。

## 维护

改依赖的流程：改 `pyproject.toml` → 同步 `docker/pip-packages.txt` 和 `docker/Dockerfile` → 重新构建镜像（同名覆盖或换名）→ 旧镜像按需删除。

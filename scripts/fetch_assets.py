"""下载竞赛数据与模型权重（一个入口，两种目标）。

仓库根目录执行：

    uv run python scripts/fetch_assets.py dataset                    # 下载并校验 Kaggle 数据
    uv run python scripts/fetch_assets.py models --list              # 看清单
    uv run python scripts/fetch_assets.py models qwen-omni-3b        # 拉基座权重
    uv run python scripts/fetch_assets.py all qwen-omni-3b qwen35-2b # 数据 + 权重
    uv run python scripts/fetch_assets.py dataset --skip-download    # 只校验

凭据：Kaggle 用 `~/.kaggle/access_token`（或环境变量 KAGGLE_API_TOKEN）；HF 走
`HF_ENDPOINT`（国内可设 https://hf-mirror.com），ModelScope 用 `modelscope` 包。
数据与权重都只留在本地，`Dataset/`、`models/` 已在 .gitignore 里。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

COMPETITION = "video-emotion-uncertainty-2026"
EXPECTED_LINES = {"train.jsonl": 3354, "val.jsonl": 666, "test_release.jsonl": 734}
EXPECTED_VIDEOS = 2974
EXPECTED_SUBMISSION_ROWS = 734

# 已跑过基线的模型；`role` 只用于打印。HF 没有的会退到 ModelScope。
MODELS = {
    "qwen-omni-3b": {"hf": "Qwen/Qwen2.5-Omni-3B", "ms": "Qwen/Qwen2.5-Omni-3B", "role": "训练基座候选，VT+VA"},
    "qwen35-2b": {"hf": "Qwen/Qwen3.5-2B", "ms": "Qwen/Qwen3.5-2B", "role": "VT 轻量对照"},
    "qwen-omni-7b": {"hf": "Qwen/Qwen2.5-Omni-7B", "ms": "Qwen/Qwen2.5-Omni-7B", "role": "VA 上限对照"},
    "qwen35-08b": {"hf": "Qwen/Qwen3.5-0.8B", "ms": "Qwen/Qwen3.5-0.8B", "role": "VT 最小对照"},
    "gemma-e2b": {"hf": "google/gemma-4-E2B-it", "ms": "", "role": "基线复现，HF 许可需先接受"},
}


def find_kaggle_cli():
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    return [sys.executable, "-m", "kaggle"]


def load_env(repo):
    """读仓库根目录的 .env；不覆盖已有环境变量，空值当作未设置。"""
    path = repo / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def kaggle_credential_source():
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "环境变量 KAGGLE_API_TOKEN（.env 或 shell）"
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists():
        return f"凭据文件 {token_file}"
    return ""


def count_lines(path):
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


# ---------------------------------------------------------------- dataset


def dataset_looks_complete(dataset_dir):
    for name, expected in EXPECTED_LINES.items():
        path = dataset_dir / name
        if not path.exists() or count_lines(path) != expected:
            return False
    if not (dataset_dir / "sample_submission.csv").exists():
        return False
    videos_dir = dataset_dir / "videos"
    return videos_dir.is_dir() and sum(1 for _ in videos_dir.rglob("*.mp4")) == EXPECTED_VIDEOS


def download_dataset(dataset_dir, force):
    command = find_kaggle_cli() + [
        "competitions", "download", "-c", COMPETITION,
        "-p", str(dataset_dir.parent), "--unzip",
    ]
    if force:
        command.append("--force")
    print("运行:", " ".join(command))
    if subprocess.run(command, check=False).returncode != 0:
        raise SystemExit(
            "下载失败。常见原因：没加入竞赛 / 没接受规则（403），"
            "或凭据不在 ~/.kaggle/access_token。"
        )


def check_videos(dataset_dir):
    missing = []
    for name in ("train.jsonl", "val.jsonl", "test_release.jsonl"):
        with (dataset_dir / name).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not (dataset_dir / record["video"]).exists():
                    missing.append((name, record["sample_id"]))
    return missing


def verify_dataset(dataset_dir):
    ok = True
    summary = {}
    for name, expected in EXPECTED_LINES.items():
        path = dataset_dir / name
        if not path.exists():
            summary[name] = "缺失"
            ok = False
            continue
        actual = count_lines(path)
        summary[name] = f"{actual} 条" + ("" if actual == expected else f"（应为 {expected}）")
        ok = ok and actual == expected

    submission = dataset_dir / "sample_submission.csv"
    if submission.exists():
        rows = count_lines(submission) - 1
        summary["sample_submission.csv"] = f"{rows} 行" + ("" if rows == EXPECTED_SUBMISSION_ROWS else f"（应为 {EXPECTED_SUBMISSION_ROWS}）")
        ok = ok and rows == EXPECTED_SUBMISSION_ROWS
    else:
        summary["sample_submission.csv"] = "缺失"
        ok = False

    videos_dir = dataset_dir / "videos"
    videos = sum(1 for _ in videos_dir.rglob("*.mp4")) if videos_dir.is_dir() else 0
    summary["videos"] = f"{videos} 个 mp4" + ("" if videos == EXPECTED_VIDEOS else f"（应为 {EXPECTED_VIDEOS}）")
    ok = ok and videos == EXPECTED_VIDEOS

    missing = check_videos(dataset_dir) if videos else []
    summary["缺失视频"] = f"{len(missing)} 条标注对不上文件"
    if missing:
        ok = False
        for split, sample_id in missing[:5]:
            summary[f"缺失视频/{split}"] = sample_id

    size_gb = sum(p.stat().st_size for p in dataset_dir.rglob("*") if p.is_file()) / 1e9
    summary["体积"] = f"{size_gb:.2f} GB"
    return ok, summary


def run_dataset(args):
    dataset_dir = Path(args.dataset_dir).resolve()
    print("数据目录:", dataset_dir)
    source = kaggle_credential_source()
    print("Kaggle 凭据:", source or "未找到，请在 .env 填 KAGGLE_API_TOKEN 或放到 ~/.kaggle/access_token")
    if not args.skip_download:
        if dataset_dir.exists() and dataset_looks_complete(dataset_dir) and not args.force:
            print("本地数据已完整，跳过下载（重下加 --force）")
        elif not source:
            raise SystemExit("没有 Kaggle 凭据，无法下载")
        else:
            download_dataset(dataset_dir, force=args.force)
    if not dataset_dir.is_dir():
        raise SystemExit(f"目录不存在：{dataset_dir}")
    ok, summary = verify_dataset(dataset_dir)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("数据校验通过" if ok else "数据校验失败")
    return 0 if ok else 1


# ---------------------------------------------------------------- models


def print_models():
    print(f"{'名称':<14}{'HF':<28}{'说明'}")
    for name, info in MODELS.items():
        print(f"{name:<14}{info['hf'] or info['ms']:<28}{info['role']}")


def dry_run(names, models_dir):
    """只查仓库体积与目标目录，不下载。"""
    from shutil import disk_usage

    from huggingface_hub import HfApi

    api = HfApi()
    free_gb = disk_usage(models_dir.anchor or ".").free / 1e9
    total_gb = 0.0
    for name in names:
        info = MODELS[name]
        repo = info["hf"] or info["ms"]
        try:
            meta = api.model_info(repo, files_metadata=True)
            size = sum((s.size or 0) for s in (meta.siblings or []))
            total_gb += size / 1e9
            print(f"{name:<14}{repo:<28}约 {size / 1e9:5.1f} GB  {info['role']}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<14}{repo:<28}体积未知（{type(exc).__name__}）  {info['role']}")
    print(f"\n合计约 {total_gb:.1f} GB，目标盘剩余 {free_gb:.1f} GB（{models_dir}）")
    return 0 if total_gb < free_gb else 1


def download_model(name, models_dir, source, revision):
    info = MODELS[name]
    target = models_dir / name
    target.mkdir(parents=True, exist_ok=True)
    if source == "modelscope":
        if not info["ms"]:
            raise SystemExit(f"{name} 没有 ModelScope 仓库，改用 --source hf")
        from modelscope import snapshot_download as ms_download

        print(f"ModelScope: {info['ms']} -> {target}")
        path = ms_download(info["ms"], local_dir=str(target), revision=revision)
    else:
        from huggingface_hub import snapshot_download

        repo = info["hf"] or info["ms"]
        print(f"Hugging Face: {repo} -> {target}")
        path = snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            revision=revision,
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.ot"],
        )
    print(f"完成: {path}")


def describe_model(models_dir, name):
    target = models_dir / name
    if not target.is_dir():
        return f"{name}: 未下载"
    files = [p for p in target.rglob("*") if p.is_file()]
    size_gb = sum(p.stat().st_size for p in files) / 1e9
    has_config = (target / "config.json").exists()
    return f"{name}: {size_gb:.2f} GB, {len(files)} 个文件" + ("" if has_config else "，缺 config.json")


def run_models(args):
    if args.list:
        print_models()
        return 0
    names = args.names or ["qwen-omni-3b"]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SystemExit(f"未知模型 {unknown}，可选：{', '.join(MODELS)}")
    models_dir = Path(args.models_dir).resolve()
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"HF endpoint: {endpoint}，HF_TOKEN: {'已设置' if os.environ.get('HF_TOKEN') else '未设置'}")
    if args.dry_run:
        return dry_run(names, models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        download_model(name, models_dir, args.source, args.revision)
    ok = True
    for name in names:
        line = describe_model(models_dir, name)
        print(" ", line)
        ok = ok and "未下载" not in line and "缺 config.json" not in line
    return 0 if ok else 1


def main():
    repo = Path(__file__).resolve().parents[1]
    load_env(repo)
    parser = argparse.ArgumentParser(description="下载竞赛数据与模型权重")
    sub = parser.add_subparsers(dest="target", required=True)

    dataset = sub.add_parser("dataset", help="下载并校验 Kaggle 竞赛数据")
    dataset.add_argument("--dataset-dir", default=str(repo / "Dataset"))
    dataset.add_argument("--skip-download", action="store_true", help="只校验")
    dataset.add_argument("--force", action="store_true", help="忽略本地已有文件重新下载")

    models = sub.add_parser("models", help="下载模型权重")
    models.add_argument("names", nargs="*", help="模型名，默认 qwen-omni-3b")
    models.add_argument("--models-dir", default=str(repo / "models"))
    models.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    models.add_argument("--revision", default=None, help="固定 commit / tag，便于复现")
    models.add_argument("--dry-run", action="store_true", help="只查体积，不下载")
    models.add_argument("--list", action="store_true", help="只打印清单")

    all_parser = sub.add_parser("all", help="数据 + 权重")
    all_parser.add_argument("names", nargs="*", help="模型名，默认 qwen-omni-3b")
    all_parser.add_argument("--dataset-dir", default=str(repo / "Dataset"))
    all_parser.add_argument("--models-dir", default=str(repo / "models"))
    all_parser.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    all_parser.add_argument("--revision", default=None)
    all_parser.add_argument("--dry-run", action="store_true", help="只查体积，不下载")
    all_parser.add_argument("--skip-download", action="store_true")

    args = parser.parse_args()
    if args.target == "dataset":
        return run_dataset(args)
    if args.target == "models":
        return run_models(args)
    args.force = False
    code = run_dataset(args)
    if code != 0:
        return code
    args.list = False
    return run_models(args)


if __name__ == "__main__":
    sys.exit(main())

"""在本地与 Intern Discovery 云盘之间同步代码。

仓库根目录执行：

    uv run python scripts/sync_code.py pack              # 打包成 dist/mm-code-<时间>.tgz
    uv run python scripts/sync_code.py pack --with-git   # 连 .git 一起带走（保留历史）
    uv run python scripts/sync_code.py upload            # 传到云盘 /MM-LLMs-Emotion/
    uv run python scripts/sync_code.py download          # 从云盘拉回当前目录

打包默认排除 Dataset/、models/、.venv/、outputs/、reference/ 这类大件，
只带代码和文档，产物一般几十 KB；数据和模型在云盘 /data/assets 放一次即可。

上传下载调用平台提供的 discovery-ctl；开发机里通常自带，本地安装方式见
docs/平台-Intern-Discovery.md。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SKIP_DIRS = {
    ".git", ".venv", ".uv-cache", ".uv-python", "__pycache__", ".agents",
    "Dataset", "models", "outputs", "logs", "evidence", "dist", "reference",
}
SKIP_SUFFIXES = {".tgz", ".tar", ".gz", ".pyc", ".zip"}
SKIP_NAMES = {"skills-lock.json"}
REMOTE_ROOT = "/MM-LLMs-Emotion"


def iter_files(repo, with_git):
    for path in sorted(repo.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(repo)
        if not with_git and rel.parts[0] == ".git":
            continue
        if any(part in SKIP_DIRS for part in rel.parts[:-1]) or rel.parts[0] in SKIP_DIRS:
            continue
        if path.suffix in SKIP_SUFFIXES or path.name in SKIP_NAMES or path.name == ".env":
            continue
        yield path, rel


def do_pack(args):
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "dist"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = out_dir / f"mm-code-{stamp}.tgz"
    count = 0
    with tarfile.open(archive, "w:gz") as tar:
        for path, rel in iter_files(repo, args.with_git):
            tar.add(path, arcname=str(rel))
            count += 1
    size_kb = archive.stat().st_size / 1024
    print(f"打包完成: {archive}  {count} 个文件  {size_kb:.0f} KB")
    print("上传: uv run python scripts/sync_code.py upload")
    return 0


def find_ctl():
    exe = shutil.which("discovery-ctl")
    if exe:
        return exe
    # 官方安装器默认落在这里，有的 shell 里 PATH 没带上
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "discovery" / "bin" / "discovery-ctl.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "discovery" / "bin" / "discovery-ctl",
        Path.home() / ".local" / "bin" / "discovery-ctl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    print("找不到 discovery-ctl。开发机里一般自带（which discovery-ctl）；"
          "本地需要按平台给的安装方式装一次，见 docs/平台-Intern-Discovery.md。")
    return None


def do_upload(args):
    exe = find_ctl()
    if not exe:
        return 1
    repo = Path(__file__).resolve().parents[1]
    archives = sorted((repo / "dist").glob("mm-code-*.tgz"))
    if not archives:
        print("dist/ 里没有打包产物，先运行 pack")
        return 1
    archive = archives[-1]
    command = [exe, "clouddisk", "upload", "--local", str(archive), "--remote-path", args.remote]
    print("运行:", " ".join(command))
    result = subprocess.run(command, check=False)
    return result.returncode


def do_download(args):
    exe = find_ctl()
    if not exe:
        return 1
    command = [exe, "clouddisk", "download", "--remote-path", args.remote, "--output", args.output]
    print("运行:", " ".join(command))
    result = subprocess.run(command, check=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="本地 <-> 云盘代码同步")
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="打包代码（默认不含大件）")
    pack.add_argument("--with-git", action="store_true", help="连 .git 一起打包")

    upload = sub.add_parser("upload", help="上传最近的打包到云盘")
    upload.add_argument("--remote", default=REMOTE_ROOT, help=f"云盘目标目录，默认 {REMOTE_ROOT}")

    download = sub.add_parser("download", help="从云盘下载")
    download.add_argument("--remote", default=REMOTE_ROOT + "/", help="云盘来源路径")
    download.add_argument("--output", default=".", help="本地落地目录")

    args = parser.parse_args()
    if args.command == "pack":
        return do_pack(args)
    if args.command == "upload":
        return do_upload(args)
    return do_download(args)


if __name__ == "__main__":
    sys.exit(main())

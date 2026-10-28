"""把课程数据转成 ms-swift 的 messages JSONL（SFT / GRPO 两种）。

用法（开发机，仓库根目录）：

    # SFT 训练集
    python scripts/to_swift.py --task sft --split train \
        --dataset Dataset/train.jsonl --output /data/assets/swift/sft_train.jsonl

    # GRPO：只要 prompt 和奖励列，不要 assistant 回答
    python scripts/to_swift.py --task grpo --split train \
        --dataset Dataset/train.jsonl --output /data/assets/swift/grpo_train.jsonl

    # 冒烟用：先转 10 条
    python scripts/to_swift.py --task grpo --dataset Dataset/train.jsonl \
        --limit 10 --output /data/assets/swift/grpo_smoke.jsonl

转换约定：

- VT 只给画面和文字（``<video>``）；VA 给画面 + 独立抽取的 16 kHz 单声道音轨
  （``<video><audio>``）。逐样本用占位符区分，不依赖全局 ``USE_AUDIO_IN_VIDEO``。
  音轨从视频里用 ffmpeg 抽出；抽不出来时补 1 秒静音，与 baseline 的兜底一致。
- GRPO 行附带 ``gold_emotion`` / ``gold_conflict`` / ``needs_clarification`` 三列，
  奖励插件通过 kwargs 读取；``needs_clarification`` 为 null 表示该样本没有补标。
- 缺失媒体的行会被跳过并计数，不会让整份数据转换中断。
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import CLASSES, SYSTEM  # noqa: E402
from trainlib import target_json  # noqa: E402


def load_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_annotations(path):
    if not path:
        return {}
    rows = load_rows(path)
    return {row["sample_id"]: row for row in rows}


def sanitize(name):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def extract_audio(video, out):
    """抽 16 kHz 单声道 wav；失败时写 1 秒静音并返回 False（与 baseline 一致）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return True
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(out),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=180)
        ok = result.returncode == 0 and out.exists() and out.stat().st_size > 0
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        out.unlink(missing_ok=True)
        sf.write(out, np.zeros(16000, dtype="float32"), 16000)
    return ok


def build_prompt(row):
    if row["protocol"] == "VT":
        text = "任务：结合视频画面和以下文字判断情感。\n文字：" + (row.get("text_content") or "")
        return "<video>" + text
    return "<video><audio>任务：结合视频画面和音频判断情感。"


def main():
    parser = argparse.ArgumentParser(description="课程数据 -> ms-swift messages JSONL")
    parser.add_argument("--task", choices=["sft", "grpo"], required=True)
    parser.add_argument("--dataset", default="Dataset/train.jsonl")
    parser.add_argument("--output", help="输出的 messages JSONL（--dry-run 时可省略）")
    parser.add_argument("--audio-dir", default="/data/assets/swift/audio")
    parser.add_argument("--annotations", help="补标 JSONL：sample_id, is_uncertain, clarifying_question")
    parser.add_argument("--protocol", choices=["all", "VT", "VA"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.output:
        parser.error("需要 --output（--dry-run 除外）")

    rows = load_rows(args.dataset)
    if args.protocol != "all":
        rows = [row for row in rows if row["protocol"] == args.protocol]
    if args.limit:
        rows = rows[:args.limit]
    annotations = load_annotations(args.annotations)
    root = Path(args.dataset).parent

    if args.dry_run:
        stats = {
            "rows": len(rows),
            "VT": sum(row["protocol"] == "VT" for row in rows),
            "VA": sum(row["protocol"] == "VA" for row in rows),
            "missing_media": sum(not (root / row["video"]).exists() for row in rows),
            "annotated": sum(row["sample_id"] in annotations for row in rows),
        }
        print(json.dumps(stats, ensure_ascii=False))
        return 0

    stats = {"rows": len(rows), "VT": 0, "VA": 0, "missing_media": 0,
             "silent_audio": 0, "annotated": 0}
    records = []
    for row in rows:
        sample_id = row["sample_id"]
        video = (root / row["video"]).resolve()
        if not video.exists():
            stats["missing_media"] += 1
            continue
        ann = annotations.get(sample_id)
        needs = None if ann is None else (1 if ann.get("is_uncertain") else 0)
        if ann is not None:
            stats["annotated"] += 1

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_prompt(row)}]
        record = {"messages": messages, "videos": [str(video)]}

        if row["protocol"] == "VT":
            stats["VT"] += 1
        else:
            stats["VA"] += 1
            wav = Path(args.audio_dir) / (sanitize(sample_id) + ".wav")
            if not extract_audio(video, wav):
                stats["silent_audio"] += 1
            record["audios"] = [str(wav)]

        if args.task == "sft":
            target = target_json(
                row,
                is_uncertain=bool(ann.get("is_uncertain")) if ann else False,
                question=(ann.get("clarifying_question") or "none") if ann else "none",
            )
            messages.append({"role": "assistant", "content": target})
        else:
            record["gold_emotion"] = row["views"]["M12"]["label"]
            record["gold_conflict"] = "yes" if row["sample_type"] == "Conflict" else "no"
            record["needs_clarification"] = needs

        records.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"写入 {len(records)} 条 -> {output}")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

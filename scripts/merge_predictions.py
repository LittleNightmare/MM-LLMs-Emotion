"""把多个分片（--shard）的 predictions.jsonl 合并成一份，并重算指标。

    uv run python scripts/merge_predictions.py --dataset Dataset/val.jsonl \
        --output outputs/val-merged \
        --predictions outputs/val-s0 --predictions outputs/val-s1 \
        --predictions outputs/val-s2 --predictions outputs/val-s3

输出与 baseline.py 一致的 predictions.jsonl 与 metrics.json；有标签时给完整指标，
无标签（测试集）时只给预测分布。重复 sample_id 以后出现的为准，缺样本会报数。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import summarize  # noqa: E402


def load_split(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def main():
    parser = argparse.ArgumentParser(description="合并分片预测并重算指标")
    parser.add_argument("--dataset", required=True, help="原始 JSONL，用来确定样本顺序与检查缺失")
    parser.add_argument("--predictions", action="append", required=True, help="分片输出目录，可重复")
    parser.add_argument("--output", required=True, help="合并后的输出目录")
    args = parser.parse_args()

    order = [row["sample_id"] for row in load_split(args.dataset)]
    merged = {}
    for directory in args.predictions:
        pred_file = Path(directory) / "predictions.jsonl"
        if not pred_file.exists():
            raise SystemExit(f"找不到 {pred_file}")
        for line in pred_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                merged[record["sample_id"]] = record

    missing = [sid for sid in order if sid not in merged]
    records = [merged[sid] for sid in order if sid in merged]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = summarize(records)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"合并 {len(records)} 条，来自 {len(args.predictions)} 个分片")
    if missing:
        print(f"缺 {len(missing)} 条，例如 {missing[:3]}")
    errors = sum("error" in r for r in records)
    print(f"错误 {errors} 条")
    if "emotion_accuracy" in metrics.get("all", {}):
        a = metrics["all"]
        print(f"准确率 {a['emotion_accuracy']:.4f}  Macro-F1 {a['emotion_macro_f1_present']:.4f}  "
              f"冲突 F1 {a['conflict_f1']:.4f}  JSON {a['json_valid_rate']:.4f}")
    else:
        print(json.dumps(metrics["all"], ensure_ascii=False))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())

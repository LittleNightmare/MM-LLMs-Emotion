"""把 predictions.jsonl 汇总成 Kaggle 提交 CSV，并校验格式。

仓库根目录执行：

    # 一个 run 直接出提交
    uv run python scripts/export_submission.py \
        --predictions outputs/qwen-omni-3b-test --test Dataset/test_release.jsonl --output submission.csv

    # VT / VA 分模型跑时传多个目录，按 sample_id 合并
    uv run python scripts/export_submission.py \
        --predictions outputs/vt-run --predictions outputs/va-run \
        --test Dataset/test_release.jsonl --output submission.csv

    # 只校验已有 CSV
    uv run python scripts/export_submission.py --check submission.csv --test Dataset/test_release.jsonl

校验四条：id 集合与 test_release 完全一致；pred_emotion 只能是三个类别；
is_uncertain 只能是 yes/no；clarifying_question 不能为空（空则补 none 并计数）。
有任何硬错误就不写 CSV，退出码非零。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

CLASSES = ("negative", "positive", "neutral")
HEADER = ["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"]


def load_test_ids(path):
    ids = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.append(json.loads(line)["sample_id"])
    return ids


def load_predictions(paths):
    """按 sample_id 合并多个 predictions.jsonl，先出现的优先。"""
    merged = {}
    for directory in paths:
        pred_file = Path(directory) / "predictions.jsonl"
        if not pred_file.exists():
            raise SystemExit(f"找不到 {pred_file}")
        with pred_file.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                merged.setdefault(record["sample_id"], record)
    return merged


def normalize(record):
    """取出提交需要的三个字段，返回 (row, problems)。"""
    problems = []
    prediction = record.get("prediction") or {}
    emotion = prediction.get("pred_emotion")
    uncertain = prediction.get("is_uncertain")
    question = prediction.get("clarifying_question")

    if emotion not in CLASSES:
        problems.append(f"pred_emotion={emotion!r}")
    if uncertain not in ("yes", "no"):
        problems.append(f"is_uncertain={uncertain!r}")
    if not isinstance(question, str) or not question.strip():
        question = "none"
        problems.append("clarifying_question 为空，已补 none")
    return [record["sample_id"], emotion, uncertain, question.strip()], problems


def build_rows(test_ids, predictions, fill_missing=False):
    rows, hard, soft = [], [], []
    missing = [i for i in test_ids if i not in predictions]
    extra = [i for i in predictions if i not in set(test_ids)]
    for sample_id in test_ids:
        if sample_id not in predictions:
            if fill_missing:
                rows.append([sample_id, "neutral", "no", "none"])
                soft.append(f"{sample_id}: 缺预测，已补 neutral/no/none")
            continue
        row, problems = normalize(predictions[sample_id])
        rows.append(row)
        for problem in problems:
            (hard if "已补" not in problem else soft).append(f"{sample_id}: {problem}")
    if missing and not fill_missing:
        hard.append(f"缺少 {len(missing)} 条预测，例如 {missing[:3]}")
    elif missing:
        soft.append(f"共 {len(missing)} 条预测缺失，已按 neutral/no/none 补齐")
    if extra:
        soft.append(f"多出 {len(extra)} 条不在测试集的预测，例如 {extra[:3]}")
    return rows, hard, soft


def check_csv(path, test_ids):
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)
    problems = []
    if header != HEADER:
        problems.append(f"表头是 {header}，应为 {HEADER}")
    ids = [r.get("sample_id") for r in rows]
    if len(rows) != len(test_ids):
        problems.append(f"{len(rows)} 行，应为 {len(test_ids)} 行")
    missing = [i for i in test_ids if i not in set(ids)]
    if missing:
        problems.append(f"缺少 {len(missing)} 个 id，例如 {missing[:3]}")
    bad_emotion = sum(r.get("pred_emotion") not in CLASSES for r in rows)
    bad_uncertain = sum(r.get("is_uncertain") not in ("yes", "no") for r in rows)
    empty_question = sum(not (r.get("clarifying_question") or "").strip() for r in rows)
    if bad_emotion:
        problems.append(f"{bad_emotion} 行 pred_emotion 非法")
    if bad_uncertain:
        problems.append(f"{bad_uncertain} 行 is_uncertain 非法")
    if empty_question:
        problems.append(f"{empty_question} 行 clarifying_question 为空")
    tuples = [[r.get("sample_id"), r.get("pred_emotion"), r.get("is_uncertain"),
               r.get("clarifying_question")] for r in rows]
    return problems, tuples


def describe(rows):
    from collections import Counter

    emotions = Counter(r[1] for r in rows)
    uncertain = sum(r[2] == "yes" for r in rows)
    questions = sum(r[3] != "none" for r in rows)
    return {"rows": len(rows), "emotions": dict(emotions),
            "uncertain_rate": round(uncertain / len(rows), 4) if rows else 0,
            "question_rate": round(questions / len(rows), 4) if rows else 0}


def main():
    parser = argparse.ArgumentParser(description="生成并校验 Kaggle 提交 CSV")
    parser.add_argument("--predictions", action="append", default=[],
                        help="含 predictions.jsonl 的目录，可重复传入")
    parser.add_argument("--test", default="Dataset/test_release.jsonl")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--check", help="只校验已有 CSV，不生成")
    parser.add_argument("--fill-missing", action="store_true",
                        help="解码失败等导致缺预测时，用 neutral/no/none 补齐，保证 734 行")
    args = parser.parse_args()

    test_ids = load_test_ids(args.test)

    if args.check:
        problems, rows = check_csv(args.check, test_ids)
        for problem in problems:
            print("问题:", problem)
        print(json.dumps(describe(rows), ensure_ascii=False))
        print("校验通过" if not problems else "校验失败")
        return 0 if not problems else 1

    if not args.predictions:
        parser.error("需要 --predictions 或 --check")
    predictions = load_predictions(args.predictions)
    rows, hard, soft = build_rows(test_ids, predictions, fill_missing=args.fill_missing)

    for note in soft:
        print("提示:", note)
    for problem in hard:
        print("错误:", problem)
    if hard:
        print("未写出 CSV，先修掉上面的错误")
        return 1

    output = Path(args.output)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(f"已写出 {output}：{len(rows)} 行")
    print(json.dumps(describe(rows), ensure_ascii=False))
    problems, _ = check_csv(output, test_ids)
    print("自检通过" if not problems else f"自检失败: {problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

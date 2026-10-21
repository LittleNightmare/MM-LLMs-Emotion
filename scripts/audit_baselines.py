"""Verify full coverage, protocol separation, and identical preprocessed media."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
parser.add_argument("--dataset", default="Dataset/val.jsonl")
parser.add_argument("--output", default="outputs/baseline-audit.json")
args = parser.parse_args()
gold = {r["sample_id"]: r for r in (
    json.loads(s) for s in Path(args.dataset).read_text(encoding="utf-8").splitlines()
)}
summary = {"expected": len(gold), "runs": {}, "media_compared": 0}
caches = []
common_settings = None
for name in args.runs:
    directory = Path(name)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    protocol = config.get("protocol", "all")
    settings = {key: config[key] for key in [
        "system_prompt", "frames", "max_side", "max_new_tokens", "dtype", "do_sample", "seed",
    ]}
    if common_settings is None:
        common_settings = settings
    assert settings == common_settings, (name, "shared inference settings differ")
    expected = {sid: row for sid, row in gold.items() if protocol == "all" or row["protocol"] == protocol}
    rows = [json.loads(s) for s in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(expected) and {r["sample_id"] for r in rows} == set(expected), name
    assert hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest() == config["dataset_sha256"]
    for row in rows:
        reference = gold[row["sample_id"]]
        assert row["gold_emotion"] == reference["views"]["M12"]["label"]
        assert row["protocol"] == reference["protocol"]
        assert row["gold_conflict"] == ("yes" if reference["sample_type"] == "Conflict" else "no")
        assert "error" not in row, (name, row["sample_id"], row.get("error"))
        has_audio = any("audio" in k or k == "input_features" for k in row["input_keys"])
        assert has_audio == (row["protocol"] == "VA"), (name, row["sample_id"])
        assert len(row["media"]["selected_indices"]) == config["frames"]
        assert row["media"]["sha256"] == hashlib.sha256(
            (Path(args.dataset).parent / reference["video"]).read_bytes()
        ).hexdigest()
    source = directory / "baseline_source.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config["script_sha256"]
    summary["runs"][name] = {
        "n": len(rows), "protocol": protocol, "protocol_checks_passed": True,
        "source_hash_verified": True, "dataset_and_original_media_hash_verified": True,
    }
    caches.append(Path(config.get("media_cache", "outputs/media_cache")))
summary["shared_settings_verified"] = common_settings
for other in caches[1:]:
    for file in caches[0].rglob("*"):
        if not file.is_file() or file.suffix not in (".mp4", ".wav"):
            continue
        peer = other / file.relative_to(caches[0])
        assert peer.exists(), str(peer)
        assert hashlib.sha256(file.read_bytes()).digest() == hashlib.sha256(peer.read_bytes()).digest(), str(peer)
        summary["media_compared"] += 1
Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

"""Offline composition using protocol only; never choose by gold or correctness."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from baseline import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vt-run", required=True)
    p.add_argument("--va-run", required=True)
    p.add_argument("--dataset", default="Dataset/val.jsonl")
    p.add_argument("--output", default="outputs/small-protocol-routing")
    args = p.parse_args()
    gold = [json.loads(s) for s in Path(args.dataset).read_text(encoding="utf-8").splitlines()]
    dataset_hash = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    sources, source_info = {}, {}
    common = None
    for protocol, run in [("VT", args.vt_run), ("VA", args.va_run)]:
        directory = Path(run)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        assert config["dataset_sha256"] == dataset_hash
        settings = {k: config[k] for k in ["system_prompt", "frames", "max_side", "max_new_tokens", "dtype", "do_sample", "seed"]}
        if common is None:
            common = settings
        assert settings == common
        file = directory / "predictions.jsonl"
        rows = [json.loads(s) for s in file.read_text(encoding="utf-8").splitlines()]
        selected = [r for r in rows if r["protocol"] == protocol]
        expected = {r["sample_id"] for r in gold if r["protocol"] == protocol}
        assert len(selected) == len(expected) and {r["sample_id"] for r in selected} == expected
        assert all("error" not in r for r in selected)
        sources[protocol] = {r["sample_id"]: r for r in selected}
        source_info[protocol] = {
            "run": run, "model_id": config["model_id"],
            "predictions_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
    combined = []
    for reference in gold:
        protocol = reference["protocol"]
        row = dict(sources[protocol][reference["sample_id"]])
        assert row["gold_emotion"] == reference["views"]["M12"]["label"]
        row["route_model"] = source_info[protocol]["model_id"]
        row["route_source"] = source_info[protocol]["run"]
        combined.append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "kind": "offline_protocol_composition", "sources": source_info,
        "routing_uses": "protocol only, never gold labels or prediction correctness",
        "selection_note": "Candidate selected after validation inspection; not an independent held-out estimate",
        "dataset_sha256": dataset_hash, "n": len(combined),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "composition_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8")
    metrics = summarize(combined)
    # Recorded timings/memory belong to the source runs, not a measured combined service.
    metrics.pop("median_seconds", None)
    metrics.pop("max_peak_allocated_gib", None)
    metrics["note"] += " Offline composition; combined deployment memory/latency was not measured."
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["all"], ensure_ascii=False))


if __name__ == "__main__":
    main()

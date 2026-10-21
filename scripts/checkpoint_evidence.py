"""Record checkpoint bytes and compare weight hashes with the official HF repo.

An HF main commit is only attributed to the downloaded weights when all weight
SHA256 values match. ModelScope's mutable 'master' alone is not a revision pin.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-metadata-file", help="Previously fetched official HF API response")
    args = parser.parse_args()
    root = Path(args.model_path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "model_id": args.model_id, "model_path": str(root),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_url": f"https://huggingface.co/{args.model_id}", "files": {},
    }
    for file in sorted(root.iterdir()):
        if file.is_file() and (file.suffix in (".json", ".safetensors", ".jinja", ".model", ".txt")
                               or file.name in ("README.md", "LICENSE", ".mv")):
            evidence["files"][file.name] = {"bytes": file.stat().st_size, "sha256": sha256(file)}
    try:
        if args.hf_metadata_file:
            info = json.loads(Path(args.hf_metadata_file).read_text(encoding="utf-8"))
        else:
            url = f"https://huggingface.co/api/models/{args.model_id}?blobs=true"
            with urllib.request.urlopen(url, timeout=45) as response:
                info = json.load(response)
        assert info["id"] == args.model_id
        (output / "huggingface_metadata.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        official = {s["lfs"]["sha256"] for s in info["siblings"]
                    if s["rfilename"].endswith(".safetensors") and s.get("lfs")}
        local = {meta["sha256"] for name, meta in evidence["files"].items()
                 if name.endswith(".safetensors")}
        evidence["hf_main_commit_observed"] = info["sha"]
        evidence["all_weight_hashes_match_official"] = bool(local) and local == official
        evidence["weight_verified_commit"] = info["sha"] if evidence["all_weight_hashes_match_official"] else None
    except Exception as exc:
        evidence["hf_metadata_error"] = str(exc)
        evidence["weight_verified_commit"] = None
    for name in ("README.md", "config.json", "generation_config.json", "chat_template.jinja", ".mv"):
        if (root / name).exists():
            (output / name).write_bytes((root / name).read_bytes())
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence.items() if k != "files"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

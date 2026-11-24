#!/usr/bin/env python3
"""Regenerate P4 aggregate CSV/JSON results from saved paired outputs, CPU-only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from assist.ft.evaluate_lora import read_jsonl, summarize, digest


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_paired_rows(rows, tests):
    """Fail closed if saved output rows do not match the frozen prompts."""
    if len(rows) != len(tests):
        raise ValueError("Paired outputs are incomplete relative to the prepared test set.")
    for i, (row, example) in enumerate(zip(rows, tests)):
        if row.get("example_id") != example["example_id"]:
            raise ValueError(f"Paired outputs are reordered or mismatched at row {i}")
        for field in ("task_id", "group_id", "domain", "op", "level"):
            if row.get(field) != example.get(field):
                raise ValueError(f"Paired output {i} differs in {field}")
        if row.get("task_input") != example["user"]:
            raise ValueError(f"Paired output {i} has mismatched task input")
        if row.get("prompt_hash") != digest(example["system"] + example["user"]):
            raise ValueError(f"Paired output {i} has mismatched prompt hash")
        for model in ("base", "lora"):
            cell = row.get(model)
            if not isinstance(cell, dict) or not {
                "cleaned_text", "compliance_flag", "leakage_flag", "latency_seconds"
            }.issubset(cell):
                raise ValueError(f"Paired output {i} lacks {model} evaluation fields")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_06b_lora_v1.yaml")
    parser.add_argument("--evaluation-dir", required=True, help="folder containing run_manifest.json and paired_outputs.jsonl")
    parser.add_argument("--output-dir", default=None, help="required: must be a separate directory to preserve original run artifacts")
    parser.add_argument("--compare-summary", default=None, help="optional recorded summary.json to compare with regenerated metrics")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(args.evaluation_dir)
    if args.output_dir is None:
        parser.error("--output-dir is required; original evaluation artifacts must not be overwritten")
    dest = Path(args.output_dir)
    if dest.resolve() == source.resolve():
        parser.error("--output-dir must differ from --evaluation-dir")
    if args.compare_summary and (dest / "summary.json").resolve() == Path(args.compare_summary).resolve():
        parser.error("--output-dir must not overwrite the recorded --compare-summary")
    manifest_path, outputs_path = source / "run_manifest.json", source / "paired_outputs.jsonl"
    if not manifest_path.exists() or not outputs_path.exists():
        raise SystemExit("Expected run_manifest.json and paired_outputs.jsonl in the evaluation directory.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if manifest.get("experiment_id") != cfg["experiment_id"]:
        raise SystemExit("Experiment ID differs from the selected config; refusing to summarize mismatched outputs.")
    if manifest.get("base_model") != cfg["base_model"] or manifest.get("base_model_revision") != cfg["base_model_revision"]:
        raise SystemExit("Pinned base model or revision differs from the evaluation manifest.")
    if manifest.get("config_sha256") != expected_config:
        raise SystemExit("Config hash differs from the evaluation manifest. Use the exact saved config used for generation.")

    prepared_path = ROOT / cfg["evaluation"]["test_file"]
    if not prepared_path.exists():
        raise SystemExit(f"Prepared test split not found: {prepared_path}")
    tests = read_jsonl(prepared_path)
    expected_ids = [row["example_id"] for row in tests]
    expected_ids_hash = sha256_text("\n".join(expected_ids))
    expected_prompts_hash = sha256_text("\n".join(row["example_id"] + row["system"] + row["user"] for row in tests))
    if manifest.get("test_ids_sha256") != expected_ids_hash or manifest.get("test_prompts_sha256") != expected_prompts_hash:
        raise SystemExit("Prepared test split does not match the recorded evaluation manifest.")
    rows = read_jsonl(outputs_path)
    found_ids = [row.get("example_id") for row in rows]
    if len(found_ids) != len(set(found_ids)):
        raise SystemExit("Duplicate example_id values in paired outputs.")
    if found_ids != expected_ids:
        raise SystemExit("Paired outputs are incomplete or out of test-set order; refusing to report a full-test summary.")
    try:
        validate_paired_rows(rows, tests)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dest.mkdir(parents=True, exist_ok=True)
    summarize(rows, cfg, dest)
    if args.compare_summary:
        old = json.loads(Path(args.compare_summary).read_text(encoding="utf-8"))
        new = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
        # The committed summary includes provenance/interpretation text that may
        # be edited without changing any statistic. Compare every computed field
        # except that free-text note; human-rating/source notes are also excluded.
        comparable_old = {key: old.get(key) for key in new if key != "interpretation_limit"}
        comparable_new = {key: new[key] for key in new if key != "interpretation_limit"}
        if comparable_old != comparable_new:
            raise SystemExit("Regenerated summary differs from the saved summary; inspect config, split and raw outputs.")
    print(json.dumps({"summary_regenerated": str(dest / "summary.json"), "paired_examples": len(rows),
                      "base_model_revision": cfg["base_model_revision"], "config_sha256": expected_config,
                      "recorded_summary_match": bool(args.compare_summary)}, indent=2))


if __name__ == "__main__":
    main()

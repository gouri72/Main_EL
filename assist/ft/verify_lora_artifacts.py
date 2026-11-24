#!/usr/bin/env python3
"""Verify the saved P4 run and optionally load the adapter for a tiny GPU smoke check."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from assist.ft.evaluate_lora import read_jsonl


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_06b_lora_v1.yaml")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--load-adapter", action="store_true", help="load base+adapter and run a four-token GPU smoke generation")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training, evaluation = Path(args.training_dir), Path(args.evaluation_dir)
    adapter = training / "adapter"
    manifest_path, outputs_path = evaluation / "run_manifest.json", evaluation / "paired_outputs.jsonl"
    if not adapter.is_dir(): raise SystemExit(f"Adapter directory is missing: {adapter}")
    if not manifest_path.is_file() or not outputs_path.is_file():
        raise SystemExit("Evaluation run_manifest.json or paired_outputs.jsonl is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_model") != cfg["base_model"] or manifest.get("base_model_revision") != cfg["base_model_revision"]:
        raise SystemExit("Evaluation manifest base model/revision differs from the selected config.")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if manifest.get("config_sha256") != config_hash:
        raise SystemExit("Evaluation manifest config hash differs from this config file.")
    recorded_hashes = manifest.get("adapter_files_sha256", {})
    current_hashes = {path.name: sha256_file(path) for path in sorted(adapter.iterdir()) if path.is_file()}
    if not current_hashes: raise SystemExit("Adapter directory contains no files.")
    if recorded_hashes and current_hashes != recorded_hashes:
        raise SystemExit("Adapter file hashes do not match the hashes recorded with evaluation.")
    test_path = ROOT / cfg["evaluation"]["test_file"]
    test_rows = read_jsonl(test_path)
    rows = read_jsonl(outputs_path)
    ids = [row.get("example_id") for row in rows]
    expected_ids = [row["example_id"] for row in test_rows]
    if len(ids) != len(set(ids)) or ids != expected_ids:
        raise SystemExit(f"Raw paired output set is incomplete or mismatched: found {len(ids)}, expected {len(expected_ids)}.")
    expected_prompts_hash = hashlib.sha256("\n".join(row["example_id"] + row["system"] + row["user"] for row in test_rows).encode("utf-8")).hexdigest()
    if manifest.get("test_prompts_sha256") != expected_prompts_hash:
        raise SystemExit("Prepared test prompts do not match the evaluation manifest.")
    result = {"adapter_dir": str(adapter), "adapter_files": current_hashes,
              "base_model": cfg["base_model"], "base_model_revision": cfg["base_model_revision"],
              "evaluation_examples": len(rows), "config_sha256": config_hash,
              "adapter_load_smoke_test": "not run"}
    if args.load_adapter:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available(): raise SystemExit("CUDA GPU is not available for --load-adapter.")
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["base_model_revision"])
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg["base_model_revision"],
                                                      torch_dtype=torch.float16, device_map="auto")
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()
        example = test_rows[0]
        messages = [{"role": "system", "content": example["system"]}, {"role": "user", "content": example["user"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        if generated.shape[1] <= inputs["input_ids"].shape[1]:
            raise SystemExit("Adapter loaded but smoke generation returned no new tokens.")
        result["adapter_load_smoke_test"] = "passed; generated four-token continuation"
        result["torch_version"] = torch.__version__
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

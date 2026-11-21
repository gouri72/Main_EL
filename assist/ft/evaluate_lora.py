#!/usr/bin/env python3
"""Paired first-pass base-vs-LoRA generation on the held-out task-group test set."""
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from assist.generate_hints import clean, tok_limit
from assist.validator import load_solutions, load_tasks, validate


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/qwen3_06b_lora_v1.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-examples", type=int, default=0, help="optional pilot cap; 0 means full test set")
    ap.add_argument("--training-dir", default=None, help="override training output location (for Drive checkpoints)")
    ap.add_argument("--output-dir", default=None, help="override evaluation output location (use Drive to resume)")
    ap.add_argument("--annotation-sample-size", type=int, default=120,
                    help="balanced blind-rating sample; 0 means include every held-out pair")
    args = ap.parse_args()
    if args.annotation_sample_size < 0:
        ap.error("--annotation-sample-size must be 0 (all) or a positive integer")
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    test_rows = read_jsonl(ROOT / cfg["evaluation"]["test_file"])
    if args.max_examples:
        test_rows = test_rows[:args.max_examples]
    training_dir = Path(args.training_dir or cfg["output_dir"])
    if not training_dir.is_absolute(): training_dir = ROOT / training_dir
    output_dir = Path(args.output_dir or cfg["evaluation"]["output_dir"])
    if not output_dir.is_absolute(): output_dir = ROOT / output_dir
    model_dir = training_dir / "adapter"
    print(json.dumps({"test_examples": len(test_rows), "distinct_task_groups": len({r['group_id'] for r in test_rows}),
                      "adapter_exists": model_dir.exists(), "config": cfg["evaluation"]}, indent=2))
    if args.dry_run:
        return
    if not model_dir.exists():
        raise SystemExit(f"LoRA adapter missing: {model_dir}. Train it first; no evaluation was run.")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Run paired generation on Colab GPU; no evaluation was run.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_id = cfg["base_model"]
    revision = cfg["base_model_revision"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                 torch_dtype=torch.float16, device_map="auto")
    paired = PeftModel.from_pretrained(base, str(model_dir))
    paired.eval()
    tasks = load_tasks(ROOT / "data/tasks.jsonl")
    solutions = load_solutions(ROOT / "data/solutions.jsonl")
    gen = cfg["evaluation"]["generation"]
    outdir = output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment_id": cfg["experiment_id"], "base_model": cfg["base_model"],
                "base_model_revision": cfg["base_model_revision"],
                "adapter_dir": str(model_dir), "generation": gen,
                "test_ids_sha256": hashlib.sha256("\n".join(r["example_id"] for r in test_rows).encode()).hexdigest(),
                "test_prompts_sha256": hashlib.sha256("\n".join(r["example_id"] + r["system"] + r["user"] for r in test_rows).encode()).hexdigest(),
                "config_sha256": file_digest(ROOT / args.config),
                "adapter_files_sha256": {p.name: file_digest(p) for p in sorted(model_dir.iterdir()) if p.is_file()},
                "test_count": len(test_rows)}
    manifest_path = outdir / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise SystemExit("Existing evaluation manifest differs. Use a new versioned output directory; refusing to mix runs.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output_path = outdir / "paired_outputs.jsonl"
    output = read_jsonl(output_path) if output_path.exists() else []
    required = {"example_id", "task_id", "group_id", "task_input", "prompt_hash", "base", "lora"}
    if any(not required.issubset(row) for row in output):
        raise SystemExit("Saved paired outputs lack required fields; use a fresh versioned evaluation directory.")
    completed = {r["example_id"] for r in output}
    if len(completed) != len(output):
        raise SystemExit("Duplicate example IDs found in saved paired outputs; refusing unsafe resume.")
    if not completed.issubset({r["example_id"] for r in test_rows}):
        raise SystemExit("Saved outputs contain IDs outside this test set; use a new output directory.")

    def run_one(row, condition):
        torch.manual_seed(int(gen["paired_example_seed"]) + int(digest(row["example_id"]), 16) % 2**31)
        torch.cuda.manual_seed_all(int(gen["paired_example_seed"]) + int(digest(row["example_id"]), 16) % 2**31)
        messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(next(paired.parameters()).device)
        adapter_context = paired.disable_adapter() if condition == "base" else nullcontext()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started = time.perf_counter()
        with adapter_context, torch.inference_mode():
            generated = paired.generate(**inputs, max_new_tokens=tok_limit(row["level"], row["domain"]),
                do_sample=gen["do_sample"], temperature=gen["temperature"], top_p=gen["top_p"],
                top_k=gen["top_k"], pad_token_id=tok.pad_token_id)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        raw = tok.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        text = clean(raw)
        verdict = validate(tasks[row["task_id"]], row["op"], row["level"], text,
                           solutions.get(row["task_id"]))
        return {"model": condition, "raw_text": raw, "cleaned_text": text,
                "latency_seconds": elapsed, "leakage_flag": verdict["leakage_flag"],
                "compliance_flag": verdict["compliance_flag"], "reasons": verdict["reasons"],
                "leak_reasons": verdict["leak_why"]}

    for i, row in enumerate(test_rows, 1):
        if row["example_id"] in completed: continue
        order = ["base", "lora"] if i % 2 else ["lora", "base"]
        by_model = {m: run_one(row, m) for m in order}
        output.append({**{k: row[k] for k in ("example_id", "task_id", "group_id", "domain", "op", "level")},
                       "task_input": row["user"],
                       "prompt_hash": digest(row["system"] + row["user"]),
                       "base": by_model["base"], "lora": by_model["lora"]})
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(output[-1], ensure_ascii=False) + "\n")
        if i % 50 == 0:
            print(f"generated and saved paired examples: {len(output)}/{len(test_rows)}", flush=True)

    summarize(output, cfg, outdir)
    make_annotation_sheet(output, outdir, int(gen["paired_example_seed"]), args.annotation_sample_size)


def summarize(rows, cfg, outdir):
    def accepted(x):
        return bool(x["compliance_flag"] and not x["leakage_flag"])
    models = {}
    for model in ("base", "lora"):
        cells = [r[model] for r in rows]
        models[model] = {"n": len(cells), "validator_pass_rate": sum(accepted(x) for x in cells) / max(1, len(cells)),
                         "validator_compliance_rate": sum(x["compliance_flag"] for x in cells) / max(1, len(cells)),
                         "leakage_rate": sum(x["leakage_flag"] for x in cells) / max(1, len(cells)),
                         "mean_generation_seconds": statistics.mean(x["latency_seconds"] for x in cells) if cells else None}
    groups = defaultdict(list)
    for r in rows: groups[r["group_id"]].append(float(accepted(r["lora"])) - float(accepted(r["base"])))
    # Resample whole task groups, while retaining the per-example weighted rate difference.
    observed = (sum(sum(v) for v in groups.values()) / sum(map(len, groups.values()))) if rows else 0.0
    rng = random.Random(cfg["evaluation"]["bootstrap_seed"])
    boot = []
    for _ in range(int(cfg["evaluation"]["bootstrap_replicates"])):
        sample_groups = [rng.choice(list(groups.values())) for _ in groups]
        sample_values = [value for group in sample_groups for value in group]
        boot.append(statistics.mean(sample_values))
    boot.sort()
    lo = boot[int(0.025 * len(boot))] if boot else None
    hi = boot[min(len(boot) - 1, int(0.975 * len(boot)))] if boot else None
    by_stratum = {}
    for key in ("domain", "level"):
        vals = defaultdict(list)
        for r in rows: vals[str(r[key])].append(r)
        for value, subset in vals.items():
            d = {m: sum(accepted(r[m]) for r in subset) / len(subset) for m in ("base", "lora")}
            by_stratum[f"{key}={value}"] = {"n": len(subset), "base_pass_rate": d["base"],
                                              "lora_pass_rate": d["lora"], "difference": d["lora"] - d["base"]}
    summary = {"experiment_id": cfg["experiment_id"], "n_paired_examples": len(rows),
               "n_task_clusters": len(groups), "models": models,
               "paired_validator_pass_rate_difference_lora_minus_base": observed,
               "task_cluster_bootstrap_95_ci": [lo, hi], "by_domain_and_level": by_stratum,
               "interpretation_limit": "These are rule-based validator and latency outcomes on base-model pseudo-label data. They do not establish hint correctness, pedagogical benefit, or human learning gains."}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metric_rows = []
    for dimension in ("overall", "domain", "level", "domain_level"):
        buckets = defaultdict(list)
        for r in rows:
            if dimension == "overall": label = "all"
            elif dimension == "domain": label = str(r["domain"])
            elif dimension == "level": label = str(r["level"])
            else: label = f"{r['domain']}_L{r['level']}"
            buckets[label].append(r)
        for label, subset in sorted(buckets.items()):
            for model in ("base", "lora"):
                n = len(subset)
                metric_rows.append({"dimension": dimension, "stratum": label, "model": model, "n": n,
                    "compliance_rate": sum(r[model]["compliance_flag"] for r in subset) / n,
                    "leakage_rate": sum(r[model]["leakage_flag"] for r in subset) / n,
                    "first_pass_acceptance": sum(accepted(r[model]) for r in subset) / n,
                    "mean_latency_seconds": statistics.mean(r[model]["latency_seconds"] for r in subset)})
    op_buckets = defaultdict(list)
    for r in rows: op_buckets[r["op"]].append(r)
    for label, subset in sorted(op_buckets.items()):
        for model in ("base", "lora"):
            n = len(subset)
            metric_rows.append({"dimension": "operation", "stratum": label, "model": model, "n": n,
                "compliance_rate": sum(r[model]["compliance_flag"] for r in subset) / n,
                "leakage_rate": sum(r[model]["leakage_flag"] for r in subset) / n,
                "first_pass_acceptance": sum(accepted(r[model]) for r in subset) / n,
                "mean_latency_seconds": statistics.mean(r[model]["latency_seconds"] for r in subset)})
    with (outdir / "comparison_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]) if metric_rows else [])
        writer.writeheader(); writer.writerows(metric_rows)
    errors = []
    for r in rows:
        for model in ("base", "lora"):
            m = r[model]
            if not accepted(m):
                categories = sorted(set(m["reasons"] + ["leakage:" + x for x in m["leak_reasons"]]))
                errors.append({"example_id": r["example_id"], "task_id": r["task_id"],
                    "group_id": r["group_id"], "domain": r["domain"], "op": r["op"], "level": r["level"],
                    "model": model, "failure_categories": ";".join(categories), "output": m["cleaned_text"]})
    with (outdir / "validator_failures.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["example_id", "task_id", "group_id", "domain", "op", "level", "model", "failure_categories", "output"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(errors)
    cases = []
    for r in rows:
        base_ok, lora_ok = accepted(r["base"]), accepted(r["lora"])
        if base_ok and lora_ok: category = "both_pass"
        elif not base_ok and lora_ok: category = "base_fails_lora_passes"
        elif base_ok and not lora_ok: category = "lora_fails_base_passes"
        else: category = "both_fail"
        cases.append({"example_id": r["example_id"], "task_id": r["task_id"], "group_id": r["group_id"],
                "domain": r["domain"], "op": r["op"], "level": r["level"], "paired_case": category,
                "base_failure_categories": ";".join(sorted(set(r["base"]["reasons"] + ["leakage:" + x for x in r["base"]["leak_reasons"]]))),
                "lora_failure_categories": ";".join(sorted(set(r["lora"]["reasons"] + ["leakage:" + x for x in r["lora"]["leak_reasons"]]))),
                "base_output": r["base"]["cleaned_text"], "lora_output": r["lora"]["cleaned_text"],
                "reviewer_explanation": ""})
    with (outdir / "paired_error_cases.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["example_id", "task_id", "group_id", "domain", "op", "level", "paired_case",
                  "base_failure_categories", "lora_failure_categories", "base_output", "lora_output", "reviewer_explanation"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(cases)
    print(json.dumps(summary, indent=2))


def make_annotation_sheet(rows, outdir, seed, sample_size):
    rng = random.Random(seed)
    if sample_size:
        strata = defaultdict(list)
        for row in rows: strata[(row["domain"], row["level"])].append(row)
        chosen = []
        per_stratum = max(1, sample_size // max(1, len(strata)))
        for key in sorted(strata):
            candidates = list(strata[key]); rng.shuffle(candidates)
            # Prefer different held-out task groups within each domain/level stratum.
            picked, used = [], set()
            for row in candidates:
                if row["group_id"] not in used:
                    picked.append(row); used.add(row["group_id"])
                    if len(picked) >= per_stratum: break
            if len(picked) < per_stratum:
                for row in candidates:
                    if row not in picked:
                        picked.append(row)
                        if len(picked) >= per_stratum: break
            chosen.extend(picked)
        rows = chosen[:sample_size]
    sheet, key = [], []
    for r in rows:
        flip = rng.choice([False, True])
        a, b = (r["lora"], r["base"]) if flip else (r["base"], r["lora"])
        rating_id = digest(r["example_id"] + str(seed))
        sheet.append({"rating_id": rating_id, "domain": r["domain"], "operation": r["op"],
                      "level": r["level"], "prompt_hash": r["prompt_hash"],
                      "task_input": r["task_input"],
                      "output_A": a["cleaned_text"],
                      "output_B": b["cleaned_text"], "A_relevance_1to5": "", "B_relevance_1to5": "",
                      "A_correctness_1to5": "", "B_correctness_1to5": "",
                      "A_level_fit_1to5": "", "B_level_fit_1to5": "", "preference_A_B_tie": "",
                      "notes": ""})
        key.append({"rating_id": rating_id, "example_id": r["example_id"], "group_id": r["group_id"],
                    "A_is": "lora" if flip else "base", "B_is": "base" if flip else "lora"})
    with (outdir / "blind_annotation_sheet.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]) if sheet else ["rating_id"]); w.writeheader(); w.writerows(sheet)
    with (outdir / "annotation_key.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]) if key else ["rating_id"]); w.writeheader(); w.writerows(key)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only dataset-quality audit for HumanFirst-Route P3.

Usage from repository root:
    python data/scripts/audit_p3_quality.py

This script never edits tasks, transfer variants, solutions, or hints.
Automated checks cannot establish semantic equivalence or real-world learning.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ("understand_plan", "execute", "debug_verify", "explain")
TITLE_RE = re.compile(r"\nAbstract \((.*?)\):\n\[1\]", re.S)
CLAIM_RE = re.compile(r"\AClaim: (.*?)\n\nAbstract \(", re.S)
FUNCTION_RE = re.compile(r"\bassert\s+([A-Za-z_]\w*)\s*\(")


def read_jsonl(path):
    records = {}
    for num, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        tid = item["task_id"]
        if tid in records:
            raise ValueError(f"duplicate task_id {tid!r} at line {num}")
        records[tid] = item
    return records


def title(task):
    match = TITLE_RE.search(task["prompt"])
    return match.group(1).strip() if match else None


def claim(task):
    match = CLAIM_RE.search(task["prompt"])
    return match.group(1).strip() if match else None


def op_names(task):
    return tuple(x["op"] for x in task["operations"])


def group_counts(values):
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def analyze(tasks):
    """Return a read-only audit dictionary. All checks are deterministic."""
    base = {k: v for k, v in tasks.items() if not k.endswith("_T")}
    variants = {k: v for k, v in tasks.items() if k.endswith("_T")}
    hard, warn = defaultdict(list), defaultdict(list)
    op_counts = defaultdict(Counter)
    pattern_counts = defaultdict(Counter)
    cross = defaultdict(lambda: defaultdict(Counter))
    base_titles, variant_titles = [], []

    for tid, task in sorted(base.items()):
        domain, difficulty = task["domain"], task["difficulty"]
        ops = op_names(task)
        pattern_counts[domain][ops] += 1
        cross[domain][difficulty][ops] += 1
        for op in ops:
            op_counts[domain][op] += 1
        pair_id = task.get("transfer_task_id")
        variant = variants.get(pair_id)
        if variant is None:
            hard["missing_variant"].append(tid)
            continue
        if variant["domain"] != domain:
            hard["domain_mismatch"].append(tid)
        if variant["difficulty"] != difficulty:
            hard["difficulty_mismatch"].append(tid)
        if op_names(variant) != ops:
            hard["operation_mismatch"].append(tid)
        if variant["prompt"] == task["prompt"]:
            hard["identical_prompts"].append(tid)
        if variant.get("transfer_task_id") != variant["task_id"]:
            hard["variant_self_reference"].append(tid)

        if domain == "coding":
            base_tests = set(task["gold"]["tests"].splitlines())
            variant_tests = set(variant["gold"]["tests"].splitlines())
            if base_tests & variant_tests:
                hard["identical_test_lines"].append(tid)
            orig_fn = FUNCTION_RE.search(task["gold"]["tests"])
            new_fn = FUNCTION_RE.search(variant["gold"]["tests"])
            if orig_fn and new_fn and orig_fn.group(1) == new_fn.group(1):
                hard["unchanged_function_name"].append(tid)
        elif domain == "science":
            bt, vt = title(task), title(variant)
            if bt is None or vt is None:
                hard["science_title_parse_failed"].append(tid)
            if claim(task) == claim(variant):
                hard["identical_science_claim"].append(tid)
            if bt == vt:
                hard["same_abstract_within_pair"].append(tid)
            if task["gold"]["label"] != variant["gold"]["label"]:
                hard["science_label_mismatch"].append(tid)
            for t in (task, variant):
                if not all(sent in t["prompt"] for sent in t["gold"]["evidence"]):
                    hard["evidence_not_in_prompt"].append(t["task_id"])
            base_titles.append((tid, bt))
            variant_titles.append((tid, vt))
        else:
            hard["unexpected_domain"].append(tid)

    expected_ids = {t["transfer_task_id"] for t in base.values()}
    for vid in sorted(set(variants) - expected_ids):
        hard["orphan_variant"].append(vid)
    base_title_set = {v for _, v in base_titles}
    for tid, t in variant_titles:
        if t in base_title_set:
            hard["variant_abstract_in_base_set"].append(tid)
    variant_counts = Counter(v for _, v in variant_titles)
    if group_counts([v for _, v in variant_titles]):
        hard["shared_variant_abstract"].extend(
            [tid for tid, t in variant_titles if variant_counts[t] > 1]
        )
    base_title_groups = defaultdict(list)
    for tid, t in base_titles:
        base_title_groups[t].append(tid)
    for t, ids in base_title_groups.items():
        if len(ids) > 1:
            warn["shared_base_abstracts"].extend(ids)
    for tid, task in sorted(base.items()):
        if len(task["prompt"]) > 6000:
            warn["long_prompt_over_6000_chars"].append(tid)
    return {
        "base": base, "variants": variants,
        "hard": dict(hard), "warn": dict(warn),
        "op_counts": op_counts, "pattern_counts": pattern_counts,
        "cross": cross, "base_title_groups": base_title_groups,
    }


def render(report):
    base, variants = report["base"], report["variants"]
    lines = [
        "# P3 automated dataset-quality audit", "",
        "This is a structural/metadata audit, **not** human annotation or semantic transfer validation.",
        "No task or hint records were modified.", "",
        "## Counts", "",
        f"- Session tasks: {len(base)}", f"- Transfer variants: {len(variants)}",
        f"- Coding session tasks: {sum(t['domain']=='coding' for t in base.values())}",
        f"- Science session tasks: {sum(t['domain']=='science' for t in base.values())}", "",
        "## Operation metadata", "",
        "| Domain | Difficulty | Tasks | Distinct operation lists | Plan | Execute | Debug/verify | Explain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in ("coding", "science"):
        for difficulty in (1, 2, 3):
            tasks = [t for t in base.values() if t["domain"] == domain and t["difficulty"] == difficulty]
            ops = Counter(o for t in tasks for o in op_names(t))
            lines.append(f"| {domain} | {difficulty} | {len(tasks)} | {len(report['cross'][domain][difficulty])} | "
                         + " | ".join(str(ops[o]) for o in OPS) + " |")
    lines += ["", "## Structural failures", "", "| Check | Count | Example IDs |", "|---|---:|---|"]
    if report["hard"]:
        for issue, ids in sorted(report["hard"].items()):
            lines.append(f"| {issue} | {len(set(ids))} | {', '.join(sorted(set(ids))[:6])} |")
    else:
        lines.append("| None detected | 0 | — |")
    lines += ["", "## Warnings (not automatic failures)", "", "| Check | Affected IDs | Example IDs |", "|---|---:|---|"]
    if report["warn"]:
        for issue, ids in sorted(report["warn"].items()):
            lines.append(f"| {issue} | {len(set(ids))} | {', '.join(sorted(set(ids))[:6])} |")
    else:
        lines.append("| None detected | 0 | — |")
    lines += [
        "", "## Interpretation and limitations", "",
        "- AST-based coding labels are deterministic heuristic proxies; they have not been validated by this script.",
        "- SciFact operation labels are heuristic except gold-grounded evidence-location (`execute`).",
        "- Coding pairs change function names and test inputs; these are near-transfer tests, not far transfer.",
        "- Matched SciFact label/evidence structure does not establish equivalent underlying reasoning skill.",
        "- Repeated session abstracts require leakage-safe sampling and train/evaluation split controls.",
        "- The P2 simulator's transfer method, not semantic correctness on these pairs, determines simulated outcomes.",
        "- The independent 40-task operation/difficulty audit remains necessary.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "stats" / "p3_quality_report.md")
    args = parser.parse_args()
    report = analyze(read_jsonl(args.tasks))
    content = render(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(content)
    print(f"Report: {args.output}")
    if report["hard"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

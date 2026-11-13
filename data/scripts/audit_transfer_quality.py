
#!/usr/bin/env python3
"""Read-only P3 transfer-pair and task-metadata audit."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks.jsonl"
OUTPUT = ROOT / "audit" / "transfer_quality_report.md"

OPS = (
    "understand_plan",
    "execute",
    "debug_verify",
    "explain",
)


def load_tasks():
    records = [
        json.loads(line)
        for line in TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(records) != len({t["task_id"] for t in records}):
        raise ValueError("Duplicate task IDs detected")

    return {t["task_id"]: t for t in records}


def science_title(task):
    match = re.search(
        r"Abstract \((.*?)\):\n",
        task["prompt"],
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else None


def science_claim(task):
    match = re.search(
        r"Claim:\s*(.*?)\n\nAbstract",
        task["prompt"],
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else None


def operations(task):
    return tuple(op["op"] for op in task["operations"])


def audit(tasks):
    base = {
        tid: t
        for tid, t in tasks.items()
        if not tid.endswith("_T")
    }

    warnings = defaultdict(list)
    stats = Counter()
    operation_counts = defaultdict(Counter)
    difficulty_ops = defaultdict(lambda: defaultdict(set))

    science_bases_by_title = defaultdict(list)
    science_variants_by_title = defaultdict(list)

    for tid, task in base.items():
        domain = task["domain"]
        difficulty = task["difficulty"]
        variant_id = task["transfer_task_id"]

        stats[f"{domain}_base"] += 1

        for op in operations(task):
            operation_counts[domain][op] += 1

        difficulty_ops[domain][difficulty].add(
            operations(task)
        )

        if variant_id not in tasks:
            warnings["missing_transfer"].append(tid)
            continue

        variant = tasks[variant_id]

        if domain != variant["domain"]:
            warnings["domain_mismatch"].append(tid)

        if difficulty != variant["difficulty"]:
            warnings["difficulty_mismatch"].append(tid)

        if operations(task) != operations(variant):
            warnings["operation_mismatch"].append(tid)

        if task["prompt"] == variant["prompt"]:
            warnings["identical_prompts"].append(tid)

        if domain == "coding":
            stats["coding_pairs"] += 1

            base_tests = set(
                task["gold"]["tests"].splitlines()
            )
            variant_tests = set(
                variant["gold"]["tests"].splitlines()
            )

            if base_tests & variant_tests:
                warnings["shared_coding_tests"].append(tid)

        elif domain == "science":
            stats["science_pairs"] += 1

            title_a = science_title(task)
            title_b = science_title(variant)

            science_bases_by_title[title_a].append(tid)
            science_variants_by_title[title_b].append(tid)

            if title_a == title_b:
                warnings["same_science_abstract"].append(tid)

            if science_claim(task) == science_claim(variant):
                warnings["same_science_claim"].append(tid)

            if (
                task["gold"]["label"]
                != variant["gold"]["label"]
            ):
                warnings["science_label_mismatch"].append(tid)

            for item in (task, variant):
                evidence = item["gold"]["evidence"]
                if not all(
                    sentence in item["prompt"]
                    for sentence in evidence
                ):
                    warnings["evidence_not_in_prompt"].append(
                        item["task_id"]
                    )

    base_titles = set(science_bases_by_title)

    for title, tids in science_variants_by_title.items():
        if title in base_titles:
            warnings["variant_uses_base_abstract"].extend(tids)

        if len(tids) > 1:
            warnings["shared_variant_abstract"].extend(tids)

    shared_base_groups = {
        title: tids
        for title, tids in science_bases_by_title.items()
        if len(tids) > 1
    }

    lines = [
        "# P3 Transfer Quality Audit",
        "",
        "This report contains automated structural checks.",
        "It is not a human semantic-validity assessment.",
        "",
        "## Dataset",
        "",
        f"- Coding base tasks: {stats['coding_base']}",
        f"- Science base tasks: {stats['science_base']}",
        f"- Coding transfer pairs: {stats['coding_pairs']}",
        f"- Science transfer pairs: {stats['science_pairs']}",
        f"- Shared science base-abstract groups: "
        f"{len(shared_base_groups)}",
        "",
        "## Operation coverage",
        "",
        "| Domain | Operation | Base tasks |",
        "|---|---|---:|",
    ]

    for domain in ("coding", "science"):
        for op in OPS:
            lines.append(
                f"| {domain} | {op} | "
                f"{operation_counts[domain][op]} |"
            )

    lines += [
        "",
        "## Operation diversity by difficulty",
        "",
        "| Domain | Difficulty | Distinct operation lists |",
        "|---|---:|---:|",
    ]

    for domain in ("coding", "science"):
        for difficulty in (1, 2, 3):
            count = len(difficulty_ops[domain][difficulty])
            lines.append(
                f"| {domain} | {difficulty} | {count} |"
            )

    lines += [
        "",
        "## Potential issues",
        "",
        "These are diagnostic flags, not automatic failures.",
        "",
        "| Check | Count | Example task IDs |",
        "|---|---:|---|",
    ]

    for name in sorted(warnings):
        ids = sorted(set(warnings[name]))
        examples = ", ".join(ids[:5])
        lines.append(
            f"| {name} | {len(ids)} | {examples} |"
        )

    if not warnings:
        lines.append(
            "| No issues flagged | 0 | - |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Coding variants mainly assess near transfer.",
        "- Equal science labels do not establish equal reasoning skill.",
        "- Shared science abstracts require leakage-safe sampling.",
        "- Operation diversity does not establish label validity.",
        "- Human semantic auditing is still required.",
        "",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    audit(load_tasks())

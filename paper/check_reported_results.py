"""Fail if headline numbers in the integrated paper contradict source summaries.

This verifies committed-transcription consistency, NOT original experiment
replication from raw records.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md_rows(name):
    records = {}
    text = (ROOT / "results" / name).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("| A_novice_v2__"):
            cols = [x.strip() for x in line.strip("|").split("|")]
            if len(cols) == 9:
                records[cols[0].replace("A_novice_v2__", "")] = (
                    float(cols[1].split()[0]), float(cols[2].split()[0])
                )
    return records


def verify():
    paper = (ROOT / "paper/manuscript.md").read_text(encoding="utf-8")
    base = md_rows("summary_A_novice_v2.md")
    abl = md_rows("summary_A_novice_v2_ablations.md")
    assert len(base) == 7 and len(abl) == 6
    base_labels = {
        "always_full": "Always Full", "always_none": "Always None",
        "fixed_human_first": "Fixed Human First",
        "rule_adaptive": "Rule Adaptive", "tabular_q": "Tabular Q",
        "linear_q": "Linear Q", "linucb": "LinUCB",
    }
    ablation_labels = {
        "tabular_q": "Full Tabular Q",
        "opblind_state": "Operation identity omitted from state",
        "opblind_caps": "Operation-specific caps removed",
        "opblind_all": "All operation controls removed",
        "performance_only": "Performance-only objective",
        "loo_transfer": "Terminal transfer reward removed",
    }
    for mapping, data in ((base_labels, base), (ablation_labels, abl)):
        for key, label in mapping.items():
            immediate, transfer = data[key]
            snippet = f"| {label} | {immediate * 100:.1f}% | {transfer * 100:.1f}% |"
            assert snippet in paper, f"manuscript P1 value differs: {snippet}"

    with (ROOT / "results/person2_cross_family_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for row in rows:
        transfer = float(row["transfer_mean"]) * 100
        assert f"{transfer:.2f}%" in paper, f"P2 transfer not recorded: {transfer}"

    p4 = json.loads((ROOT / "results/qwen3_06b_lora_v1/evaluation/summary.json").read_text())
    for model in ("base", "lora"):
        pass_rate = p4["models"][model]["validator_pass_rate"] * 100
        assert f"{pass_rate:.2f}%" in paper
    assert p4["n_paired_examples"] == 618
    assert p4["human_ratings_status"] == "Not conducted; no human ratings are claimed or reported."
    assert "were **not conducted**" in paper


if __name__ == "__main__":
    verify()
    print("PAPER EVIDENCE CHECK PASSED")

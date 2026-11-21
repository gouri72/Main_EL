#!/usr/bin/env python3
"""Create PPT-ready comparison and level-compliance/leakage charts after evaluation."""
import argparse
import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/qwen3_06b_lora_v1.yaml")
    ap.add_argument("--evaluation-dir", default=None, help="override path to saved evaluation outputs")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    out = Path(args.evaluation_dir or cfg["evaluation"]["output_dir"])
    if not out.is_absolute(): out = ROOT / out
    metrics_path = out / "comparison_metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}; run paired evaluation first.")
    import matplotlib.pyplot as plt
    with metrics_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out_fig = out / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    overall = [r for r in rows if r["dimension"] == "overall"]
    labels = ["Validator acceptance", "Compliance", "Leakage flagged"]
    metrics = ["first_pass_acceptance", "compliance_rate", "leakage_rate"]
    colors = {"base": "#52677d", "lora": "#e28b35"}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.34
    for j, model in enumerate(("base", "lora")):
        row = next(r for r in overall if r["model"] == model)
        vals = [float(row[m]) * 100 for m in metrics]
        ax.bar([i + (j - 0.5) * width for i in range(3)], vals, width, label=model, color=colors[model])
    ax.set_xticks(range(3), labels); ax.set_ylim(0, 100); ax.set_ylabel("Rate (%)")
    ax.set_title("Base Qwen vs LoRA Qwen: held-out first-pass outputs")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(out_fig / "base_vs_lora_overall.png", dpi=200); plt.close(fig)

    levels = sorted({int(r["stratum"]) for r in rows if r["dimension"] == "level"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ("compliance_rate", "leakage_rate"),
                                 ("Level compliance", "Leakage flagged")):
        for j, model in enumerate(("base", "lora")):
            subset = {int(r["stratum"]): float(r[metric]) * 100 for r in rows
                      if r["dimension"] == "level" and r["model"] == model}
            ax.bar([l + (j - 0.5) * width for l in levels], [subset[l] for l in levels], width,
                   label=model, color=colors[model])
        ax.set_title(title); ax.set_xlabel("Assistance level"); ax.set_xticks(levels, [f"L{l}" for l in levels])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Rate (%)"); axes[0].legend(frameon=False)
    fig.suptitle("First-pass outcomes by assistance level"); fig.tight_layout()
    fig.savefig(out_fig / "level_compliance_leakage.png", dpi=200); plt.close(fig)
    print(f"Wrote charts under {out_fig}")


if __name__ == "__main__":
    main()

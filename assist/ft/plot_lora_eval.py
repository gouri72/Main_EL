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
    labels = ["Validator acceptance", "Compliance"]
    metrics = ["first_pass_acceptance", "compliance_rate"]
    colors = {"base": "#52677d", "lora": "#e28b35"}
    fig, (ax, leak_ax) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    width = 0.34
    for j, model in enumerate(("base", "lora")):
        row = next(r for r in overall if r["model"] == model)
        vals = [float(row[m]) * 100 for m in metrics]
        positions = [i + (j - 0.5) * width for i in range(2)]
        bars = ax.bar(positions, vals, width, label=model, color=colors[model])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
        leak = float(row["leakage_rate"]) * 100
        leak_bar = leak_ax.bar([j], [leak], width, label=model, color=colors[model])
        leak_ax.text(j, leak + 0.006, f"{leak:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(2), labels); ax.set_ylim(98, 100.15); ax.set_ylabel("Rate (%)")
    ax.set_title("Acceptance and compliance (98–100% scale)")
    ax.grid(axis="y", alpha=.2)
    leak_ax.set_xticks([0, 1], ["Base", "LoRA"]); leak_ax.set_ylim(0, .22); leak_ax.set_ylabel("Rate (%)")
    leak_ax.set_title("Leakage flagged (0–0.22% scale)"); leak_ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False); fig.suptitle("Base Qwen vs LoRA Qwen: held-out first-pass outputs")
    fig.tight_layout()
    fig.savefig(out_fig / "base_vs_lora_overall.png", dpi=200); plt.close(fig)

    levels = sorted({int(r["stratum"]) for r in rows if r["dimension"] == "level"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, metric, title, ylim in zip(axes, ("compliance_rate", "leakage_rate"),
                                       ("Level compliance (95–100% scale)", "Leakage flagged (0–2% scale)"),
                                       ((95, 100.4), (0, 2))):
        for j, model in enumerate(("base", "lora")):
            subset = {int(r["stratum"]): float(r[metric]) * 100 for r in rows
                      if r["dimension"] == "level" and r["model"] == model}
            positions = [l + (j - 0.5) * width for l in levels]
            vals = [subset[l] for l in levels]
            bars = ax.bar(positions, vals, width, label=model, color=colors[model])
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, val + (0.03 if metric == "compliance_rate" else 0.04),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=90 if metric == "leakage_rate" else 0)
        ax.set_title(title); ax.set_xlabel("Assistance level"); ax.set_xticks(levels, [f"L{l}" for l in levels])
        ax.set_ylim(*ylim); ax.set_ylabel("Rate (%)"); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("First-pass outcomes by assistance level"); fig.tight_layout()
    fig.savefig(out_fig / "level_compliance_leakage.png", dpi=200); plt.close(fig)
    print(f"Wrote charts under {out_fig}")


if __name__ == "__main__":
    main()

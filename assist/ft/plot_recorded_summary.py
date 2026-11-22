#!/usr/bin/env python3
"""Render presentation-ready SVG figures from the committed aggregate summary.

This uses only Python's standard library. It does not recreate per-example
statistics or replace plot_lora_eval.py, which uses the full evaluation CSV.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def esc(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def svg_start(width, height, title, subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="36" y="42" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#172b4d">{esc(title)}</text>',
            f'<text x="36" y="70" font-family="Arial,sans-serif" font-size="13" fill="#52677d">{esc(subtitle)}</text>']


def text(x, y, value, size=13, color="#263746", anchor="start", weight="400"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{esc(value)}</text>'


def overall_figure(summary):
    width, height = 880, 460
    lines = svg_start(width, height, "Base Qwen vs LoRA Qwen", "Held-out first-pass outputs • 618 paired examples • rule-based validator")
    base, lora = summary["models"]["base"], summary["models"]["lora"]
    metrics = [("Validator acceptance", "validator_pass_rate"), ("Level compliance", "validator_compliance_rate"), ("Leakage flagged", "leakage_rate")]
    chart_left, chart_top, chart_width, chart_height = 270, 115, 500, 245
    for tick in (0, 25, 50, 75, 100):
        x = chart_left + chart_width * tick / 100
        lines.append(f'<line x1="{x:.1f}" y1="{chart_top}" x2="{x:.1f}" y2="{chart_top+chart_height}" stroke="#dbe2e8"/>')
        lines.append(text(x, chart_top + chart_height + 22, f"{tick}%", 11, "#52677d", "middle"))
    for i, (label, key) in enumerate(metrics):
        y = chart_top + 34 + i * 78
        lines.append(text(chart_left - 14, y + 18, label, 14, "#263746", "end", "700"))
        for j, (name, record, color) in enumerate((("Base", base, "#52677d"), ("LoRA", lora, "#e28b35"))):
            val = record[key] * 100
            yy = y + j * 26
            barw = max(1, chart_width * val / 100)
            lines.append(f'<rect x="{chart_left}" y="{yy}" width="{barw:.2f}" height="16" rx="3" fill="{color}"/>')
            label_x = min(chart_left + chart_width - 3, chart_left + barw + 7)
            lines.append(text(label_x, yy + 13, f"{name}: {val:.2f}%", 11, color, "start", "700"))
    ci = summary["task_cluster_bootstrap_95_ci"]
    delta = summary["paired_validator_pass_rate_difference_lora_minus_base"] * 100
    lines.append(text(36, 410, f"Acceptance difference (LoRA − base): {delta:+.2f} percentage points", 14, "#172b4d", weight="700"))
    lines.append(text(36, 434, f"Task-cluster bootstrap 95% interval: {ci[0]*100:+.2f} to {ci[1]*100:+.2f} points; includes zero.", 12, "#52677d"))
    lines.append(text(36, 452, "Rule-based pass rates do not measure hint correctness, teaching quality, or human learning.", 11, "#8a3c2f"))
    lines.append("</svg>")
    return "\n".join(lines)


def strata_figure(summary):
    width, height = 880, 490
    lines = svg_start(width, height, "Validator acceptance by evaluation stratum", "Base and LoRA first-pass acceptance • reported held-out aggregate rates")
    groups = [("Coding", "domain=coding"), ("Science", "domain=science"), ("Level 1", "level=1"), ("Level 2", "level=2"), ("Level 3", "level=3"), ("Level 4", "level=4")]
    x0, y0, chartw, chart_h = 230, 115, 500, 300
    for tick in (0, 25, 50, 75, 100):
        x = x0 + chartw * tick / 100
        lines.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+chart_h}" stroke="#dbe2e8"/>')
        lines.append(text(x, y0 + chart_h + 22, f"{tick}%", 11, "#52677d", "middle"))
    colors = {"base": "#52677d", "lora": "#e28b35"}
    for i, (label, key) in enumerate(groups):
        row = summary["by_domain_and_level"][key]
        y = y0 + 25 + i * 47
        lines.append(text(x0 - 14, y + 18, label, 13, "#263746", "end", "700"))
        n = row["n"]
        for j, model in enumerate(("base", "lora")):
            val = row[f"{model}_pass_rate"] * 100
            yy = y + j * 18
            barw = max(1, chartw * val / 100)
            lines.append(f'<rect x="{x0}" y="{yy}" width="{barw:.2f}" height="12" rx="2" fill="{colors[model]}"/>')
            label_x = min(x0 + chartw - 4, x0 + barw + 6)
            lines.append(text(label_x, yy + 11, f"{val:.2f}%  (n={n})", 10, colors[model], "start", "700"))
    lines.append(text(36, 465, "Stratum sizes vary; these descriptive slices are not independent confirmation of an effect.", 12, "#52677d"))
    lines.append("</svg>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/qwen3_06b_lora_v1/evaluation/summary.json")
    parser.add_argument("--output-dir", default="results/qwen3_06b_lora_v1/figures")
    args = parser.parse_args()
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "base_vs_lora_overall.svg").write_text(overall_figure(summary), encoding="utf-8")
    (out / "acceptance_by_stratum.svg").write_text(strata_figure(summary), encoding="utf-8")
    print(f"Wrote two SVG figures to {out}")


if __name__ == "__main__":
    main()

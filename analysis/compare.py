#!/usr/bin/env python3
"""Compare experiments: mean +- 95% CI over seeds, and paired differences against a reference experiment.

  python analysis/compare.py --ref A_novice__always_full "results/A_novice__*" --k 20 --out results/summary_A_novice.md
Unit of analysis = per-seed mean over the last k training episodes (seeds are independent). CI = t-interval over seeds.
Paired = same seed in both experiments (paired t-test). Immediate success comes from steps.csv.
efficiency = mean transfer gain / mean assistance spent (per seed); '-' when no assistance was spent.
"""
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["immediate_success", "post_transfer", "transfer_gain", "assist_spent", "full_assist_rate", "first_attempt_rate", "takeover_rate", "efficiency"]


def per_seed(d, k):
    S = pd.read_csv(Path(d) / "steps.csv", comment="#")
    E = pd.read_csv(Path(d) / "episodes.csv", comment="#")
    e = E[E.episode >= E.episode.max() - k + 1]
    s = S[S.episode >= S.episode.max() - k + 1]
    g = e.groupby("seed")
    out = pd.DataFrame({
        "post_transfer": g.post_transfer.mean(), "transfer_gain": g.transfer_gain.mean(),
        "assist_spent": g.assist_spent.mean(), "full_assist_rate": g.full_assist_rate.mean(),
        "first_attempt_rate": g.first_attempt_rate.mean(), "takeover_rate": g.takeover_rate.mean()})
    out["immediate_success"] = s.groupby("seed").success.apply(lambda x: x.astype(float).mean())
    out["efficiency"] = out.transfer_gain / out.assist_spent.replace(0, np.nan)
    return out


def mean_ci(x, conf=0.95):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) < 2:
        return x.mean(), np.nan
    return x.mean(), stats.sem(x) * stats.t.ppf((1 + conf) / 2, len(x) - 1)


def fmt(m, h, nd=3):
    return "-" if np.isnan(m) else (f"{m:.{nd}f}" if np.isnan(h) else f"{m:.{nd}f} ± {h:.{nd}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result folders or globs")
    ap.add_argument("--ref", required=True, help="exp_id of the reference experiment for paired comparisons")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dirs = sorted({p for pat in a.dirs for p in glob.glob(pat) if (Path(p) / "episodes.csv").exists()})
    data = {Path(d).name: per_seed(d, a.k) for d in dirs}
    if a.ref not in data:
        raise SystemExit(f"reference {a.ref!r} not among {sorted(data)[:5]}...")
    lines = [f"# Results (last {a.k} episodes per seed; mean ± 95% CI over {len(data[a.ref])} seeds)", "",
             "| experiment | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for name, df in data.items():
        lines.append(f"| {name} | " + " | ".join(fmt(*mean_ci(df[m])) for m in METRICS) + " |")
    lines += ["", f"# Paired differences vs {a.ref} (experiment minus reference; paired by seed; p = paired t-test)", "",
              "| experiment | Δ immediate_success | Δ transfer_gain | p | Δ post_transfer | Δ assist_spent |", "|---|---|---|---|---|---|"]
    ref = data[a.ref]
    for name, df in data.items():
        if name == a.ref:
            continue
        cells = []
        p_gain = np.nan
        for m in ("immediate_success", "transfer_gain", "post_transfer", "assist_spent"):
            idx = df.index.intersection(ref.index)
            d = (df.loc[idx, m] - ref.loc[idx, m]).dropna()
            cells.append(fmt(*mean_ci(d)))
            if m == "transfer_gain" and len(d) > 1 and d.std() > 0:
                p_gain = stats.ttest_rel(df.loc[d.index, m], ref.loc[d.index, m]).pvalue
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {'-' if np.isnan(p_gain) else f'{p_gain:.3f}'} | {cells[2]} | {cells[3]} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
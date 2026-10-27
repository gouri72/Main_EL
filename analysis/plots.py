#!/usr/bin/env python3
"""Paper figures from results/<exp_id>/{steps,episodes}.csv. Only reads the agreed schema (see run_experiment.py).
  python analysis/plots.py results/smoke_001 [results/other_exp ...]   -> figures in paper/figs/
Figures: fading (level vs task index), pareto (immediate success vs transfer), transfer_bar (mean + 95% CI over seeds).
Per-seed means are the unit of analysis (seeds are independent), CI = t-interval over seeds."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "paper" / "figs"

def load(exp_dirs):
    S = pd.concat([pd.read_csv(Path(d) / "steps.csv", comment="#") for d in exp_dirs])
    E = pd.concat([pd.read_csv(Path(d) / "episodes.csv", comment="#") for d in exp_dirs])
    return S, E

def mean_ci(x, conf=0.95):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 2: return (x.mean() if len(x) else np.nan, np.nan)
    h = stats.sem(x) * stats.t.ppf((1 + conf) / 2, len(x) - 1)
    return x.mean(), h

def last_k(df, k=10):
    """Evaluate on the final k training episodes of each seed (policy has converged)."""
    return df[df.episode >= df.episode.max() - k + 1]

def fading(S, k=10):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for pol, g in last_k(S, k).groupby("policy"):
        per_seed = g.groupby(["seed", "task_idx"])["level"].mean().unstack()
        m = per_seed.mean(); ci = per_seed.apply(lambda c: mean_ci(c)[1])
        ax.errorbar(m.index + 1, m.values, yerr=ci.values, marker="o", capsize=2, label=pol)
    ax.set_xlabel("task index within session"); ax.set_ylabel("assistance level (0-4)"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fading.pdf"); plt.close(fig)

def pareto(S, E, k=10):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    succ = last_k(S, k).groupby(["policy", "seed"]).success.apply(lambda s: s.astype(float).mean()).groupby("policy")
    trans = last_k(E, k).groupby(["policy", "seed"]).post_transfer.mean().groupby("policy")
    for pol in succ.groups:
        sx, sy = mean_ci(succ.get_group(pol)), mean_ci(trans.get_group(pol))
        ax.errorbar(sx[0], sy[0], xerr=sx[1], yerr=sy[1], fmt="o", capsize=2, label=pol)
    ax.set_xlabel("immediate success"); ax.set_ylabel("unaided transfer success"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "pareto.pdf"); plt.close(fig)

def transfer_bar(E, k=10):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    rows = []
    for pol, g in last_k(E, k).groupby("policy"):
        m, h = mean_ci(g.groupby("seed").post_transfer.mean()); rows.append((pol, m, h))
    ax.bar([r[0] for r in rows], [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=3)
    ax.set_ylabel("unaided transfer success"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(OUT / "transfer_bar.pdf"); plt.close(fig)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    S, E = load(sys.argv[1:])
    fading(S); pareto(S, E); transfer_bar(E)
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
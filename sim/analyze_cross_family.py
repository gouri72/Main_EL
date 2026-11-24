import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("results/person2_cross_family.csv")
OUT = Path("results/person2_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

metrics = [
    "transfer",
    "practice_success",
    "reward",
    "mean_time",
    "mean_level",
    "human_first_rate",
    "frustration_rate",
    "abandonment_rate",
    "copying_rate",
]

# First aggregate episodes within each training seed
seed_data = (
    df.groupby(
        ["heldout_family", "method", "training_seed"],
        as_index=False,
    )[metrics]
    .mean()
)

seed_data.to_csv(
    OUT / "seed_level_results.csv",
    index=False,
)

# Aggregate seed-level results and calculate 95% CI
summary_rows = []

for (family, method), group in seed_data.groupby(
    ["heldout_family", "method"]
):

    row = {
        "heldout_family": family,
        "method": method,
        "n_seeds": len(group),
    }

    for metric in metrics:
        values = group[metric].dropna().to_numpy()

        mean = np.mean(values)

        if len(values) > 1:
            se = np.std(values, ddof=1) / np.sqrt(len(values))
            ci = 2.364624251 * se
        else:
            ci = 0.0

        row[f"{metric}_mean"] = mean
        row[f"{metric}_ci95"] = ci

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT / "cross_family_summary.csv",
    index=False,
)

print("\nSEED-LEVEL SUMMARY")
print(
    summary[
        [
            "heldout_family",
            "method",
            "transfer_mean",
            "transfer_ci95",
            "practice_success_mean",
            "practice_success_ci95",
            "reward_mean",
            "reward_ci95",
        ]
    ].round(3).to_string(index=False)
)

# Plot transfer comparison
methods = [
    "RL",
    "Rule_Adaptive",
    "Always_None",
    "Fixed_Human_First",
]

families = ["A", "B", "C"]

for metric in ["transfer", "practice_success", "reward"]:

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in methods:

        values = []
        errors = []

        for family in families:

            row = summary[
                (summary["heldout_family"] == family)
                & (summary["method"] == method)
            ]

            values.append(
                row[f"{metric}_mean"].iloc[0]
            )

            errors.append(
                row[f"{metric}_ci95"].iloc[0]
            )

        ax.errorbar(
            families,
            values,
            yerr=errors,
            marker="o",
            capsize=4,
            label=method,
        )

    ax.set_xlabel("Held-out Family")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"Cross-Family {metric.replace('_', ' ').title()}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUT / f"{metric}_comparison.png",
        dpi=300,
    )

    plt.close(fig)

print(f"\nAnalysis saved to: {OUT}")
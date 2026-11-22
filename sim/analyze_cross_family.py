import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

INPUT = Path("results/person2_cross_family.csv")
OUT = Path("results/person2_cross_family_summary.csv")
FIG = Path("results/person2_figures/cross_family_transfer.png")

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

summary = []

for family, group in df.groupby("heldout_family"):
    row = {
        "heldout_family": family,
        "training_families": group["training_families"].iloc[0],
        "episodes": len(group),
    }

    for metric in metrics:
        values = group[metric].dropna()
        n = len(values)
        mean = values.mean()
        sd = values.std(ddof=1)
        ci = 1.96 * sd / np.sqrt(n) if n > 1 else 0

        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
        row[f"{metric}_ci95"] = ci

    summary.append(row)

result = pd.DataFrame(summary).sort_values("heldout_family")
result.to_csv(OUT, index=False)

print("\nCross-family statistical summary")
for _, row in result.iterrows():
    print(
        f"\nFamily {row['heldout_family']} "
        f"(train {row['training_families']}):"
    )
    print(
        f"  Transfer: {row['transfer_mean']:.3f} "
        f"+/- {row['transfer_ci95']:.3f} (95% CI)"
    )
    print(
        f"  Practice: {row['practice_success_mean']:.3f} "
        f"+/- {row['practice_success_ci95']:.3f}"
    )
    print(
        f"  Reward: {row['reward_mean']:.3f} "
        f"+/- {row['reward_ci95']:.3f}"
    )
    print(f"  Mean assistance: {row['mean_level_mean']:.3f}")
    print(f"  Human-first rate: {row['human_first_rate_mean']:.3f}")

FIG.parent.mkdir(parents=True, exist_ok=True)

families = result["heldout_family"].tolist()
means = result["transfer_mean"].to_numpy()
cis = result["transfer_ci95"].to_numpy()

plt.figure(figsize=(7, 5))
plt.bar(families, means, yerr=cis, capsize=5)
plt.ylabel("Unaided transfer success")
plt.xlabel("Held-out simulator family")
plt.title("Frozen-policy cross-family generalization")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(FIG, dpi=300)
plt.close()

print(f"\nSaved summary: {OUT}")
print(f"Saved figure: {FIG}")

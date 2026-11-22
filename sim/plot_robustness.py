import csv
from pathlib import Path
import matplotlib.pyplot as plt

INPUT = Path("results/person2_robustness_summary.csv")
OUT = Path("results/person2_figures")
OUT.mkdir(parents=True, exist_ok=True)

with INPUT.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

families = ["A", "B", "C"]
profiles = ["novice", "advanced", "hint-dependent", "frustration-prone"]
levels = [0, 2, 4]


def select(family, profile=None, level=None):
    return [
        r for r in rows
        if r["family"] == family
        and (profile is None or r["profile"] == profile)
        and (level is None or int(r["level"]) == level)
    ]


def aggregate(family, level, metric):
    values = [
        float(r[f"{metric}_mean"])
        for r in select(family, level=level)
    ]
    return sum(values) / len(values)


# Figure 1: transfer performance by family and assistance level
plt.figure()

for family in families:
    values = [aggregate(family, level, "transfer_success") for level in levels]
    plt.plot(levels, values, marker="o", label=f"Family {family}")

plt.xlabel("Assistance level")
plt.ylabel("Mean simulated unaided transfer success")
plt.xticks(levels)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "transfer_by_family.png", dpi=300)
plt.close()


# Figure 2: practice success
plt.figure()

for family in families:
    values = [aggregate(family, level, "practice_success_rate") for level in levels]
    plt.plot(levels, values, marker="o", label=f"Family {family}")

plt.xlabel("Assistance level")
plt.ylabel("Mean practice success rate")
plt.xticks(levels)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "practice_success_by_family.png", dpi=300)
plt.close()


# Figure 3: profile-specific transfer
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, family in zip(axes, families):
    for profile in profiles:
        values = []

        for level in levels:
            selected = select(family, profile, level)
            values.append(
                sum(float(r["transfer_success_mean"]) for r in selected)
                / len(selected)
            )

        ax.plot(levels, values, marker="o", label=profile)

    ax.set_title(f"Family {family}")
    ax.set_xlabel("Assistance level")
    ax.set_xticks(levels)
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Mean simulated transfer success")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    fontsize=9,
)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT / "profile_transfer_comparison.png", dpi=300)
plt.close(fig)

print("Saved 3 figures in:", OUT)

import pandas as pd

# Load the steps data from our recent run
df = pd.read_csv("results/smoke_005_fixed/steps.csv", comment="#")

# We only care about the evaluation phase where the policy is frozen
eval_df = df[df["phase"] == "eval"]

if eval_df.empty:
    print("No evaluation data found. Check your phases.")
else:
    # Calculate how often HUMAN_FIRST (route 0) was chosen, grouped by domain and operation
    summary = eval_df.groupby(["domain", "op"]).agg(
        total_tasks=("route", "count"),
        human_first_pct=("route", lambda x: (x == 0).mean() * 100),
        avg_level=("level", "mean")
    ).reset_index()

    summary["human_first_pct"] = summary["human_first_pct"].round(1).astype(str) + "%"
    summary["avg_level"] = summary["avg_level"].round(2)

    print("\n--- Learned Routing Behavior (Evaluation Phase) ---")
    print(summary.to_string(index=False))
# P1 — RL benchmark and ablation reproducibility

## Verified scope

The **v2 family-A / novice** benchmark uses eight training seeds, 1,000
training episodes and 100 frozen-policy evaluation episodes **per seed**,
with eight task steps per episode. Source and configuration are in
`run_experiment.py`, `rl/policy.py` and `configs/grid/A_novice_v2__*.yaml`.

The committed summary tables are:

- `results/summary_A_novice_v2.md` — seven policies.
- `results/summary_A_novice_v2_ablations.md` — five ablations against tabular Q.

Full `results/A_novice_v2__*/steps.csv` and `episodes.csv` files were
generated locally; **they are not committed**. Reproduce them before claiming
to have independently recalculated the published summary.

## Read-only integrity checks

```powershell
python -m pytest -q tests/test_p1_policy_fixes.py tests/test_rl_compare_eval.py tests/test_rl_compare_integrity.py
python data/scripts/freeze_manifest.py --verify
```

To reproduce experiment outputs, use the exact committed YAML for each
configuration. Running the experiment runner **overwrites** its corresponding
result directory; back up local results first.

```powershell
python run_experiment.py configs/grid/A_novice_v2__tabular_q.yaml
python analysis/compare.py --ref A_novice_v2__tabular_q "results/A_novice_v2__*" --k 100 --out results/summary_A_novice_v2.md
```

The comparison utility now rejects incomplete evaluation seed runs, duplicate
episode/step identifiers, and seed-set mismatches; a temporary two-episode
smoke test cannot silently enter a 100-episode benchmark.

## Findings and interpretation

For family-A novices, observed post-transfer performance was 28.0% for
Linear Q, 27.4% for Always None, 26.6% for Rule Adaptive and 25.5%
for Tabular Q. An operation-blind-state ablation reached 28.6%, but
its immediate success was lower. The operation-cap ablation reduced
post-transfer from 25.5% to 23.7%.

The results demonstrate trade-offs **under the simulator assumptions**;
they do not establish human learning gains or that RL wins on all metrics.
Statistical comparisons are exploratory and the reported paired p-values are
not adjusted for multiple comparisons. Linear-Q outcomes vary across seeds.
The transfer measure is generated from simulator latent dynamics, **not**
execution or semantic scoring of transfer tasks.

Training and evaluation use separate episodes drawn from the **same base-task
pool**, so this does not establish generalization to entirely unseen tasks.
P2's cross-family experiment is separate and should not be pooled with v2.
Simulator A explicitly penalizes high-level assistance; the resulting
association between assistance and transfer is not causal evidence about
real students.

## Research close-out

- [x] Run and summarize seven policies
- [x] Evaluate five ablations
- [x] Analyze assistance routes and training-seed variability locally
- [x] Add frozen-evaluation integrity guards
- [ ] Archive or regenerate full raw experiment CSVs for independent reproduction
- [ ] Run and preserve full integrated CI checks after all team merges
- [ ] Integrate P1, P2, P3 and P4 results into a unified manuscript

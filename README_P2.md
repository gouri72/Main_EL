# Person 2 – Cross-Family Evaluation

## Objective

Evaluate the generalization of the simulated learner policy across different task families using a cross-family held-out evaluation protocol.

The objective is to investigate whether a policy trained on two task families can generalize to an unseen third family.

## Experimental Setup

- Task families: A, B and C
- Learner profile: Novice
- Training seeds: 8 independent seeds
- Training episodes: 120 episodes per training family
- Training families: 2 families per fold
- Held-out evaluation family: 1 family per fold
- Evaluation episodes: 50 per seed per method
- Tasks per session: 8
- Transfer repetitions: 10

## Cross-Family Evaluation

Three held-out folds were evaluated:

| Fold | Training Families | Held-Out Family |
|---|---|---|
| 1 | B, C | A |
| 2 | A, C | B |
| 3 | A, B | C |

The RL policy was trained on the two available families and evaluated on the held-out family without further training updates.

Eight training seeds were used for each fold.

## Compared Methods

Four methods were evaluated:

1. RL – Tabular Q-learning policy
2. Rule Adaptive – rule-based adaptive assistance
3. Always None – no assistance baseline
4. Fixed Human-First – fixed Human-First assistance baseline

## Evaluation Metrics

- Transfer success
- Practice success
- Total reward
- Mean time
- Mean assistance level
- Human-First rate
- Frustration rate
- Abandonment rate
- Copying rate

## Evaluation Scale

The evaluation consisted of:

- 3 held-out folds
- 8 training seeds per fold
- 4 evaluation methods
- 50 episodes per seed per method

Total evaluation episodes:

3 × 8 × 4 × 50 = 4,800

## Results

The seed-level results were aggregated across evaluation episodes within each seed.

The results were then summarized across the 8 training seeds for each held-out family and evaluation method.

| Held-Out Family | RL Transfer | Rule Adaptive | Always None | Fixed Human-First |
|---|---:|---:|---:|---:|
| A | 0.240 | 0.213 | 0.269 | 0.316 |
| B | 0.287 | 0.291 | 0.272 | 0.321 |
| C | 0.305 | 0.323 | 0.266 | 0.325 |

## Initial Findings

- Fixed Human-First achieved the highest transfer success in all three held-out families.
- RL performed better than Always None in Families B and C.
- RL did not consistently outperform the Rule Adaptive baseline.
- Cross-family performance varied depending on the held-out task family.
- The results do not establish that the RL policy outperforms the evaluated baselines.

## Statistical Analysis

Results were aggregated at the training-seed level to reduce the risk of treating individual episodes as independent training runs.

Mean performance and confidence intervals were calculated across training seeds.

The statistical analysis and plots are available in:

`results/person2_analysis/`

## Figures

- `transfer_comparison.png`
- `practice_success_comparison.png`
- `reward_comparison.png`

## Output Files

- `results/person2_cross_family.csv`
- `results/person2_analysis/seed_level_results.csv`
- `results/person2_analysis/cross_family_summary.csv`
- `results/person2_analysis/transfer_comparison.png`
- `results/person2_analysis/practice_success_comparison.png`
- `results/person2_analysis/reward_comparison.png`

## Reproducibility

Run the cross-family experiment:

```bash
python -m sim.cross_family
# Evidence inventory and final research handoff

This ledger distinguishes **committed raw evidence**, **committed summaries**, and **external or unperformed evidence**. It must not be interpreted as independently reproducing all experiments.

| Workstream | Committed evidence | External/unperformed evidence |
|---|---|---|
| P1 benchmark and ablations | `results/summary_A_novice_v2.md`, `results/summary_A_novice_v2_ablations.md`, 12 versioned configuration YAMLs, runner and policy code | Original twelve-run P1 v2 ZIP generated and verified on the user's local workstation; not uploaded to GitHub |
| P2 simulation | `sim/cross_family.py`, `sim/robustness.py`, `results/person2_cross_family.csv`, `results/person2_robustness.csv` and their summaries | Independent repetitions of trained policies across folds; baseline comparisons |
| P3 corpus | Task/solution/hint files and frozen SHA-256 manifest | Independent annotation by two human raters; semantics of transfer pairs |
| P3 exploratory audit | `data/audit/annotations_llm_assisted.csv` and `data/audit/llm_assisted_report.md` | No real-human label agreement; model-only ratings must be labeled as such |
| P4 LoRA | Pinned configurations, split manifest, training metadata, transcribed `evaluation/summary.json`, charts | Full paired first-pass outputs and adapter in team Drive; blinded human quality ratings |

## Claim-to-source map

- 600 frozen task records and operation frequencies: `data/stats/dataset_stats.md`.
- P1 28.0% Linear Q / 25.5% Tabular Q simulated post-transfer: `results/summary_A_novice_v2.md`.
- P1 operation-blind-state 28.6%, cap-removed 23.7%: `results/summary_A_novice_v2_ablations.md`.
- P2 held-out A/B/C transfer means 24.225/29.55/31.20%: `results/person2_cross_family_summary.csv`.
- P3 40-item model–heuristic audit: `data/audit/llm_assisted_report.md`.
- P4 618 paired first-pass model outputs, 99.19/99.51% rule acceptance: `results/qwen3_06b_lora_v1/evaluation/summary.json`.
- P4 model weights/raw outputs: external, not independently verified by the committed repository.

## Final release checklist

- [x] Python tests run successfully in GitHub Actions
- [x] All six frozen data artifacts match SHA-256 manifest
- [x] P1 benchmark and ablation code/configurations and summaries integrated
- [x] P2 robustness/cross-family source and summary evidence integrated
- [x] P3 model-only audit documented without fabricating human reliability
- [x] P4 automated LoRA experiment documented without asserting semantic improvement
- [x] Unified paper-ready draft composed, with provenance and limitations
- [ ] Team confirms authorship, bibliography, target venue and institutional submission requirements
- [x] Original 12-run P1 ZIP archived locally; checksum verified: `e69d1d520f294d06a395cab8a277480848f22f7dcef87bfbe0c218a37c19a236`
- [ ] Independently verify a backup of that ZIP in team-managed storage
- [ ] Verify access and hashes for P4 original Colab paired outputs and model adapter
- [ ] Obtain actual P3 independent human labels and P4 blinded hint quality judgments, **or explicitly report these as not conducted**
- [ ] Repeat P2 independently trained held-out-family folds if strong generalization claims are desired
- [ ] Confirm figures and final manuscript formatting for submission

Do not change frozen assets, fabricate results or regenerate aggregate numbers from assumptions.

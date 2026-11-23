# HumanFirst-Route: Operation-Aware Reinforcement Learning for Adaptive AI Assistance and Simulated Unaided Transfer

**Integrated research manuscript draft — 23 November 2026.** Team approval, author metadata, and venue-specific formatting are still required. This project reports simulator and automated model-evaluation results; no learner-participant experiment or independent human annotation study has been completed.

## Abstract

We evaluate **HumanFirst-Route**, an assistance router that chooses HUMAN_FIRST or AI_FIRST support and an assistance level from task-operation and observable learner-state features. A frozen benchmark has 300 base tasks and 300 paired variants from MBPP-derived programming and SciFact-derived claim verification. A separate 4,060-entry assistance bank provides levels of guidance. Seven routing policies and five ablations were assessed in a novice learner simulator with eight random seeds, 1,000 training episodes and 100 frozen-policy evaluation episodes per seed. Linear Q-learning achieved 28.0% simulated unaided post-transfer success versus 25.5% for Tabular Q-learning and 26.6% for Rule Adaptive, though Rule Adaptive achieved higher immediate task success (51.1% versus 40.6% for Linear Q). Removing operation identity from the tabular state increased simulated transfer to 28.6%, while removing operation assistance caps reduced it to 23.7%. Preliminary cross-family simulations and a separate base-versus-LoRA Qwen3-0.6B study further describe sensitivity and model rule compliance. Fine-tuned Qwen attained 99.51% rule-validator acceptance versus 99.19% for the base model across 618 paired held-out outputs, a difference whose task-cluster-bootstrap 95% interval crosses zero. These findings describe trade-offs under the **specified simulation assumptions**, not demonstrated educational effects in real students.

**Keywords:** simulated learners; reinforcement learning; assistance routing; learning transfer; operation-aware scaffolding; Qwen3-0.6B.

## 1. Introduction

Higher success while receiving AI support does not necessarily imply higher success without it. The system investigates when assistance should be offered—before an attempt or only after one—and how strongly, while distinguishing immediate completion from a simulated measure of later independent performance. We investigate (RQ1) policy trade-offs, (RQ2) effects of operation-aware state, caps and reward terms, (RQ3) sensitivity to learner-family assumptions, and (RQ4) automated hint-compliance changes following LoRA fine-tuning. All research questions concern synthetic/evaluated-system behavior, not demonstrated human learning gains.

## 2. Dataset and operation labels (P3)

The 600 task records comprise **150 coding base tasks + 150 coding paired variants** and **150 science base tasks + 150 science paired variants**. MBPP-derived programming items retain reference solutions/tests; coding variants mostly change function names and integer inputs while preserving the core algorithm, representing **near transfer**. SciFact-derived claims retain labeled rationale/evidence; their paired variants match label and evidence structure, not necessarily meaning or skill equivalence. Base-task difficulty bins 1/2/3 each contain 50 items per domain, by construction.

Operation labels cover understand/plan, execute, debug/verify and explain. Coding base membership counts are 150/150/80/74; science counts are 150/150/111/150 respectively, with overlapping memberships. Coding operations largely derive from reference-solution AST heuristics; scientific execute is evidence-grounded, while other operations and difficulty estimates are mainly heuristic. The frozen hint bank contains **4,060 guidance entries**, primarily Qwen-generated with a small handwritten and fallback component.

The 40-item GPT-6 exploratory **model-only** audit contains 20 coding and 20 science tasks. Against frozen heuristic labels, coding debug/verify agreement was 45%, explain agreement 65%, and difficulty-bin exact agreement 45%. Science debug/verify agreement was 80% and difficulty-bin exact agreement 15%. These are **model–heuristic agreement rates**, not accuracy, two-human agreement, or gold semantic validity. The proposed two-independent-human validation is unperformed.

Within-session sampling prevents duplicate science abstracts, but abstract-disjoint train/test partitions are not demonstrated for the RL study. Data provenance and weaknesses: [dataset methods](datasets.md), [P3 audit](p3_results.md), and [model-only audit report](../data/audit/llm_assisted_report.md).

## 3. Architecture and methodology

### 3.1 Task interaction and state

The agent chooses one of ten actions: HUMAN_FIRST with assistance level 0–4 or AI_FIRST with level 0–4. For HUMAN_FIRST, the simulated learner attempts the task unaided and receives assistance only if needed and permitted; AI_FIRST makes assistance available immediately. Operation-specific caps constrain available levels. The model observes domain, operation, binned knowledge estimates, difficulty and bounded history features. True simulator mastery and dependence do **not** appear in policy observations. The requested assistance level and assistance actually delivered must be distinguished.

### 3.2 Learner families and outcome definition

Simulator Family A makes high assistance levels reduce the simulated later transfer probability; Family B removes this direct damage; Family C includes practice and forgetting. Completion and transfer outcomes are stochastic draws from simulated latent variables. **No coding transfer problem is executed and no scientific response is semantically graded** to compute the reported transfer scores: task pairs provide an experimental scaffold, while transfer outcomes are a simulator-based proxy. These embedded assumptions constrain causal or pedagogical interpretation.

### 3.3 Policies and rewards

The policy comparison includes Tabular Q-learning, Linear Q-learning, LinUCB (contextual-bandit comparison), Always Full, Always None, Fixed Human First and Rule Adaptive. The configurable reward combines immediate task progress, assistance and takeover penalties, and a terminal simulated transfer difference. Ablations suppress operation identity in the policy state, remove operation caps, remove all operation controls, use a performance-only objective, or omit the terminal transfer reward. Fixed strategies do not train. Policy masking prevents invalid actions; training and evaluation episodes are labeled separately in CSVs.

### 3.4 Experimental protocols and statistics

**P1:** family-A novice setting, eight seeds, 1,000 training plus 100 frozen-policy evaluation episodes per seed, eight tasks per episode. Seed-level means and nominal paired tests compare policies. Reported 95% intervals are over eight seed-level means; multiple hypothesis tests were not corrected. Both phases sample from the same base-task corpus, so this is not unseen-task generalization.

**P2:** each cross-family fold trains a Tabular Q model for 120 episodes per training family on two families and freezes it for 50 episodes on the third family. Each held-out family has **one trained model**; its interval reflects evaluation episodes rather than independently repeated training. A separate robustness study samples four learner profiles, three families and different fixed assistance levels.

**P4:** pinned Qwen3-0.6B was fine-tuned for three LoRA epochs (rank 8, alpha 16) on 2,822 examples from 208 training tasks, with 599 development examples from 46 tasks. Paired first-pass inference compares base and LoRA on 618 held-out examples clustered into 41 task groups, using matched prompting/decoding. Fine-tuning targets are mainly the base model's filtered generated hints, so this is primarily self-distillation, not expert-corrected supervision. The adapter and raw paired outputs remain in team Drive; committed metric summaries were transcribed from team-provided Colab output.

## 4. Results

### 4.1 P1 policy benchmark

| Policy | Immediate success | Simulated unaided post-transfer | Normalized transfer gain (%) | Mean assistance spent/session |
|---|---:|---:|---:|---:|
| Always Full | 50.9% | 10.7% | -19.5 | 22.538 |
| Always None | 15.8% | 27.4% | +3.1 | 0.000 |
| Fixed Human First | 58.8% | 15.4% | -13.1 | 18.621 |
| Rule Adaptive | 51.1% | 26.6% | +1.9 | 15.692 |
| Tabular Q | 35.7% | 25.5% | +0.5 | 8.462 |
| Linear Q | 40.6% | 28.0% | +3.9 | 10.364 |
| LinUCB | 47.8% | 20.7% | -5.9 | 12.586 |

Fixed Human First yields the highest immediate success (58.8%) yet only 15.4% simulated post-transfer. Linear Q leads the seven policies in post-transfer (28.0%) but is more seed-sensitive, while Rule Adaptive attains much higher immediate success than Linear Q (51.1% versus 40.6%). Always None has competitive transfer but very low immediate completion. There is no uniformly superior policy. Transfer gain is **100 × (post_transfer − pre_transfer) / (1 − pre_transfer)**, computed per eligible episode and averaged. It is a normalized percentage measure, **not a percentage-point difference**; it should not be confused with a gap between immediate-success and post-transfer percentages. Exact intervals and seed-paired exploratory tests are in the [committed benchmark summary](../results/summary_A_novice_v2.md). Raw P1 per-step/per-episode outputs have not been uploaded to GitHub.

### 4.2 P1 operation and reward ablations

| Policy variant | Immediate success | Post-transfer | Normalized transfer gain (%) |
|---|---:|---:|---:|
| Full Tabular Q | 35.7% | 25.5% | +0.5 |
| Operation identity omitted from state | 32.3% | 28.6% | +4.7 |
| Operation-specific caps removed | 37.3% | 23.7% | -1.8 |
| All operation controls removed | 37.4% | 24.0% | -1.5 |
| Performance-only objective | 42.6% | 22.9% | -3.0 |
| Terminal transfer reward removed | 35.4% | 25.3% | +0.3 |

Removing operation identity **from the state only** retains caps/penalties; the resulting model increases transfer while lowering immediate success. Removing caps decreases simulated transfer to 23.7%. Removing the transfer reward has little observed effect (nominal paired p=0.766). The results are **mixed**, not evidence that every operation-aware component helps. See [ablation summary](../results/summary_A_novice_v2_ablations.md).

### 4.3 P2 preliminary held-out-family performance

| Held-out family | Training families | Evaluation episodes | Simulated post-transfer | Practice success |
|---|---|---:|---:|---:|
| A | B + C | 50 | 24.23% | 38.0% |
| B | A + C | 50 | 29.55% | 37.25% |
| C | A + B | 50 | 31.20% | 37.25% |

These are single-training-run fold results and should not be pooled with P1's eight-seed family-A benchmark. See [P2 committed results](../results/person2_cross_family_summary.csv).

### 4.4 P4 base-versus-LoRA model outputs

| Rule-based metric | Base Qwen | LoRA Qwen |
|---|---:|---:|
| Rule-validator acceptance | 99.19% | 99.51% |
| Rule-validator compliance | 99.35% | 99.51% |
| Rule-validator leakage flags | 0.16% | 0.00% |
| Mean generation latency | 1.53 s | 2.62 s |

The acceptance difference is **+0.32 percentage points** with a task-cluster bootstrap 95% interval of **-0.73 to +1.27 pp**, spanning zero. Thus no acceptance improvement is established. The adapter was approximately 72% slower on average. Near-ceiling rule acceptance is not a measure of factual correctness, pedagogical helpfulness or learner performance. Human blinded relevance/correctness/level-fit ratings have **not been completed**. See [recorded P4 results](../results/qwen3_06b_lora_v1/RESULTS.md).

## 5. Discussion

The simulations illustrate different routes to high immediate performance and high modeled independent performance. Under Family A, high-level assistance imposes transfer damage **by design**; it follows that assistance-heavy strategies may show lower simulated transfer. That association must not be reported as independent causal evidence about real students. The contrast between operation-blind-state Tabular Q and cap-removed Tabular Q suggests that explicit state features and action constraints play distinct roles. Reduced tabular state sparsity and decreased high-level assistance are plausible explanations for the state-ablation result, but not established causes. The strong rule-based baseline demonstrates that learning is not automatically preferable to a simple adaptive strategy.

P4 provides a separate automated hint-generation result. It is not connected to simulated outcomes because the P1/P2 learner model responds to assistance levels rather than the meaning of generated Qwen hints. Accordingly, no end-to-end fine-tuning-to-transfer gain is demonstrated.

## 6. Validity, ethics and limitations

- All outcomes on simulated learners are proxies defined by the simulator; **no human-participant improvement has been measured**.
- P1's evaluation consists of frozen policies on new episodes but the same underlying base-task pool; P2's held-out axis is family, not task identity.
- Coding variants are near-transfer, while SciFact variants match evidence/label structure without guaranteed conceptual equivalence.
- Heuristic operation and difficulty labels lack independent human validation; the 40-task GPT-6 audit is model–heuristic comparison, not human agreement.
- Multiple exploratory P1 contrasts are unadjusted and involve eight training seeds; P2 reports only one trained model per held-out fold.
- P4 self-distills mostly model-generated hints, and automatic acceptance checks cannot establish factual correctness. No blinded human-quality study has been completed.
- P1 raw evaluation CSVs are local; P4 raw pairs and adapter weights are external to GitHub, so original summary replication remains partly dependent on team-owned records.
- Any future real-learner study requires appropriate participant safeguards and institutional approvals as applicable; none is asserted here.

## 7. Conclusion

HumanFirst-Route is a functioning research prototype for comparing human-first versus AI-first routing and assistance intensity under operation constraints. Across simulator-based novice benchmarks, policies trade immediate simulated task success against modeled unaided transfer; operation-specific assistance caps and operation identity in a tabular state exhibit different effects. P4's automated Qwen3-0.6B LoRA evaluation does not show a statistically established acceptance advantage. The most important next validation tasks are repeated independently trained cross-family experiments, independently rated hint/label quality, raw-artifact archiving and—if educational effectiveness is claimed—appropriately designed real-learner assessment.

## Reproducibility and references

Repository: https://github.com/gouri72/Main_EL. P1: [README_P1](../README_P1.md); P2: [README_P2](../README_P2.md); P3: [README_P3](../README_P3.md); P4: [README_P4_FT](../README_P4_FT.md). [Evidence inventory](evidence_inventory.md).

1. Austin, J., et al. (2021). *Program Synthesis with Large Language Models*. https://arxiv.org/abs/2108.07732
2. Wadden, D., et al. (2020). *Fact or Fiction: Verifying Scientific Claims*. EMNLP. https://aclanthology.org/2020.emnlp-main.609/

*Authors must review, format, and approve this draft before external submission.*

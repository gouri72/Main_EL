# P3 automated dataset-quality audit

This is a structural/metadata audit, **not** human annotation or semantic transfer validation.
No task or hint records were modified.

## Counts

- Session tasks: 300
- Transfer variants: 300
- Coding session tasks: 150
- Science session tasks: 150

## Operation metadata

| Domain | Difficulty | Tasks | Distinct operation lists | Plan | Execute | Debug/verify | Explain |
|---|---:|---:|---:|---:|---:|---:|---:|
| coding | 1 | 50 | 4 | 50 | 50 | 12 | 7 |
| coding | 2 | 50 | 4 | 50 | 50 | 24 | 29 |
| coding | 3 | 50 | 4 | 50 | 50 | 44 | 38 |
| science | 1 | 50 | 2 | 50 | 50 | 20 | 50 |
| science | 2 | 50 | 2 | 50 | 50 | 41 | 50 |
| science | 3 | 50 | 1 | 50 | 50 | 50 | 50 |

## Structural failures

| Check | Count | Example IDs |
|---|---:|---|
| None detected | 0 | — |

## Warnings (not automatic failures)

| Check | Affected IDs | Example IDs |
|---|---:|---|
| long_prompt_over_6000_chars | 1 | sci_134 |
| shared_base_abstracts | 51 | sci_001, sci_004, sci_005, sci_014, sci_019, sci_021 |

## Interpretation and limitations

- AST-based coding labels are deterministic heuristic proxies; they have not been validated by this script.
- SciFact operation labels are heuristic except gold-grounded evidence-location (`execute`).
- Coding pairs change function names and test inputs; these are near-transfer tests, not far transfer.
- Matched SciFact label/evidence structure does not establish equivalent underlying reasoning skill.
- Repeated session abstracts require leakage-safe sampling and train/evaluation split controls.
- The P2 simulator's transfer method, not semantic correctness on these pairs, determines simulated outcomes.
- The independent 40-task operation/difficulty audit remains necessary.

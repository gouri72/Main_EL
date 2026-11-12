# P3 dataset and transfer-pair benchmarking (verified structural results)

**Status:** automated engineering checks complete; independent human
annotation and inter-annotator agreement pending.

## Dataset composition

| Metric | Coding (MBPP-derived) | Science (SciFact-derived) |
|---|---:|---:|
| Session tasks | 150 | 150 |
| Transfer variants | 150 | 150 |
| Base difficulty 1 / 2 / 3 | 50 / 50 / 50 | 50 / 50 / 50 |
| Base tasks: understand_plan | 150 | 150 |
| Base tasks: execute | 150 | 150 |
| Base tasks: debug_verify | 80 | 111 |
| Base tasks: explain | 74 | 150 |
| Distinct operation lists at difficulty 1 | 4 | 2 |
| Distinct operation lists at difficulty 2 | 4 | 2 |
| Distinct operation lists at difficulty 3 | 4 | 1 |

The 300 base tasks each have a linked transfer variant, producing 600
task records. The dataset contains 75 SUPPORT and 75 CONTRADICT
science base claims. The task-and-transfer dataset uses the frozen JSONL
schema defined in `data/tasks.jsonl`.

## Automated verification

The local integrated run on 12 November 2026 reported:

- `python data/validate_tasks.py`: 600 task records; OK.
- `python data/scripts/data_checks.py`: OK; 300 base tasks, 300 transfer
  variants, 4,060 hints with zero missing and zero extra combinations.
- `python data/scripts/audit_p3_quality.py`: zero structural failures.
- `python -m pytest -q tests/test_p3_quality.py tests/test_p3_freeze.py`:
  9 tests passed.
- `python tests/test_ops_and_sessions.py`: 6 checks passed.
- `python tests/test_data.py`: 5 checks passed.
- `python data/scripts/freeze_manifest.py --verify`: six artifacts
  matched their locally generated SHA-256 freeze manifest.

This is a transcription of actual locally shared command output, **not**
an independently rerun result in this GitHub branch.

## Diagnostic warnings and interpretation

- `sci_134` has a prompt over 6,000 characters, which can stress a
  generation model's prompt budget; this is not a schema failure.
- 51 science base IDs belong to repeated-abstract groups, corresponding
  to 23 shared abstract groups; 28 is the number of duplicated occurrences
  beyond one per group. These counts use different denominators.
- Session sampling avoids repeating an abstract **within one session**.
  The checks do not establish abstract-disjoint training/evaluation splits.
- 96 of 150 coding tasks happen to match the old difficulty-only
  label-count pattern, but the currently stored coding operation labels
  are derived from reference-solution AST features rather than difficulty
  alone.
- Coding variants principally test near transfer (changed function names
  and inputs). SciFact variants are selected by label/evidence structure;
  matching these does not guarantee semantic skill equivalence.

## What these numbers do *not* demonstrate

Zero structural failures is **not** a measurement of human annotator
agreement, operation-label semantic validity, real learner progress,
superior RL routing, or actual correctness on transfer questions.

In the current simulator, transfer outcomes are generated from latent
mastery/dependence dynamics; the paired transfer tasks are not executed
or semantically scored by the simulator.

## Open research validation

Two independent people must annotate the fixed 40-task sample without
access to heuristic labels. The annotation CLI, frozen rubric, full-sheet
validation and agreement calculator are implemented, but the human
ratings and kappa results are not yet available. Until then, report
label validity as *pending*, not as verified.

## Reproduction

See `README_P3.md`, `data/stats/p3_quality_report.md`,
`data/audit/annotation_protocol.md`, and
`data/scripts/freeze_manifest.py`.

Do not rerun dataset or hint-generation scripts for this audit;
those can overwrite the frozen data or Qwen bank.

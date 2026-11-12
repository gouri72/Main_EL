# P3 — Data, operation metadata, transfer integrity, and reproducibility

## Status

- 150 MBPP base problems and 150 coding near-transfer variants.
- 150 SciFact base claims and 150 paired evidence-based variants.
- 600 records total, with difficulty 1/2/3 balanced within base domains.
- Coding operation labels derive from reference-solution AST features; SciFact
  uses gold-grounded evidence-location and heuristic operation labels.
- Existing task/solution/hint files are frozen for policy and simulator compatibility.
- Human agreement **has not been measured**; the 40-item annotation sample
  and independent-label protocol are ready.

## Read-only verification from the repository root

```powershell
python data/validate_tasks.py
python data/scripts/data_checks.py
python data/scripts/audit_p3_quality.py
python -m pytest -q tests/test_p3_quality.py tests/test_p3_freeze.py
python tests/test_ops_and_sessions.py
python tests/test_data.py
python assist/check_bank.py
```

`data/stats/p3_quality_report.md` is the generated structural-audit report.
The audit does not claim conceptual similarity or human-label validity.

## Freeze the artifacts

After all checks pass and after confirming the six data files are the agreed
versions, create the reproducibility manifest once:

```powershell
python data/scripts/freeze_manifest.py
python data/scripts/freeze_manifest.py --verify
```

Commit `data/stats/p3_freeze_manifest.json` with the team-approved frozen
dataset. `--verify` later fails if any frozen file has changed.

This intentionally includes `data/hints.jsonl` because changes in P4 hints
would alter the data contract for downstream experiments. Human annotation
sheets are intentionally excluded.

## Interactive, independent annotation

The Python CLI presents the 40 frozen audit questions without revealing the
dataset's heuristic operation labels. Each rated task is saved immediately;
both annotators can work separately and resume without data loss.

```powershell
# First annotator: existing blank CSV
python data/scripts/annotate_tasks.py --sheet data/audit/annotations_P3.csv

# Second annotator: create their own sheet once, then begin
python data/scripts/annotate_tasks.py --sheet data/audit/annotations_second.csv --init
python data/scripts/annotate_tasks.py --sheet data/audit/annotations_second.csv

# Progress only, no rating input
python data/scripts/annotate_tasks.py --status
python data/scripts/annotate_tasks.py --sheet data/audit/annotations_second.csv --status

# Focused regression tests
python -m pytest -q tests/test_p3_annotation_cli.py
```

Keep the completed independent sheets separate until both are frozen.
No code can legitimately fill human-authored judgments on the annotators' behalf.
Do not claim that agreement analysis has been completed before both sheets
are independently populated.

## Independent annotation (not yet completed)

See `data/audit/annotation_protocol.md` and
`data/audit/audit_items.txt`. Two independent human annotators must
complete the 40-task fixed sample; automated tools must not invent ratings.

```powershell
python data/scripts/audit_agreement.py data/audit/annotations_P3.csv data/audit/annotations_second.csv
```

The agreement report is generated only when both complete sheets exist.

## Important: do not accidentally rebuild the research corpus

The scripts below are **regeneration tools** and will overwrite task files or
hints. Do not run them as part of routine verification:

- `data/scripts/build_all.py` (without `--skip-hints`)
- `data/scripts/build_coding.py`
- `data/scripts/build_science.py`
- `data/scripts/repair_science_variants.py`
- `assist/build_stub_hints.py`
- `data/scripts/audit_sample.py`

Never stage unrelated `results/` directories with `git add .`.

## Paper and limitations

See `paper/datasets.md`, `paper/related_work_p3.md`, and
`paper/p3_audit_handoff.md`. Structural validation is not a measure of
transfer-learning effectiveness. P2 simulator transfer is calculated from its
latent state rather than executing/semantic-scoring paired transfer tasks.
Also, repeated SciFact base abstracts require care in evaluation splits.

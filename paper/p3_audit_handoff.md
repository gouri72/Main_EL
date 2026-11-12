# P3 quality audit — implementation handoff

## Scope and data integrity

The added `data/scripts/audit_p3_quality.py` performs automated metadata and transfer-pair checks and writes `data/stats/p3_quality_report.md`. It does not modify `data/tasks.jsonl`, `data/solutions.jsonl`, generated hints, or annotation sheets. Its checks establish structural consistency and flag possible leakage, not semantic validity.

The updated `data/scripts/audit_agreement.py` requires two independent **completed** annotation sheets with exactly the IDs in `data/audit/sample_ids.json`. It rejects blanks, duplicate IDs, missing or extra tasks, mismatched domains, and invalid label values. It never imputes human judgments.

## Reproduction from repository root

```powershell
python data/scripts/audit_p3_quality.py
python -m pytest -q tests/test_p3_quality.py
python data/validate_tasks.py
python data/scripts/data_checks.py
python tests/test_ops_and_sessions.py
python tests/test_data.py
```

After two independent annotators have finished their sheets:

```powershell
python data/scripts/audit_agreement.py data/audit/annotations_P3.csv data/audit/annotations_second.csv
```

The quality report is generated locally and is not committed in this change. Verify actual results before using statistics in a manuscript.

## Known scientific limitations

- Coding transfer pairs mainly change function names and integer test inputs: near rather than far transfer.
- SciFact pairs share labels and evidence characteristics but may test different biomedical concepts.
- Repeated science base abstracts require leakage-aware sampling and train/test design.
- AST-derived coding operation labels and all difficulty bins are heuristic pending human validation.
- Simulator transfer outcomes are sampled by P2's model, not obtained by executing transfer problems or grading SciFact answers.
- Neither structural checks nor synthetic-learner results establish real human learning improvements.

## Paper tasks

Update `paper/datasets.md` to the audited coding counts (80 debug/verify; 74 explain), remove its obsolete claim that coding operations are deterministic functions of difficulty, and describe the human agreement report as pending until annotation sheets are completed.

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

The checked-in quality report was generated from the frozen dataset after the previous P3 integration. Regenerate and compare it when data files change; this branch does not regenerate datasets.

## Known scientific limitations

- Coding transfer pairs mainly change function names and integer test inputs: near rather than far transfer.
- SciFact pairs share labels and evidence characteristics but may test different biomedical concepts.
- Repeated science base abstracts require leakage-aware sampling and train/test design.
- AST-derived coding operation labels and all difficulty bins are heuristic pending human validation.
- Simulator transfer outcomes are sampled by P2's model, not obtained by executing transfer problems or grading SciFact answers.
- Neither structural checks nor synthetic-learner results establish real human learning improvements.

## Completion and remaining actions

The paper's counts and AST-method wording have been updated (coding: 80 debug/verify and 74 explain). The dataset sources and transfer limitations are documented in `paper/related_work_p3.md`, and the independent rating rubric is prepared in `data/audit/annotation_protocol.md`.

Run `python data/scripts/freeze_manifest.py` once to create `data/stats/p3_freeze_manifest.json`, and then `python data/scripts/freeze_manifest.py --verify` to check for subsequent modifications. Commit the generated manifest only after the team confirms which task and hint versions are frozen.

**Outstanding human-only step:** two independent annotators complete the same 40-task sheets, followed by `audit_agreement.py`; no agreement statistics should be claimed before then.

**Other-team research issue:** P1/P2 should check abstract-disjoint train/evaluation conditions and whether the transfer simulation is meaningfully linked to actual transfer-task difficulty. These are not automatic P3 data fixes.

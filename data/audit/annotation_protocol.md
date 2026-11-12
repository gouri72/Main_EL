# P3 annotation protocol (prepared; human ratings pending)

## Fixed sample and blinding

The 40 task IDs in `data/audit/sample_ids.json` define the fixed audit sample. Read
`data/audit/audit_items.txt` rather than consulting the dataset's heuristic
`operations` or `difficulty` fields. Do not rerun `audit_sample.py` while
the sample is in use: it could overwrite the sample or template.

Two people independently complete sheets with exactly these columns:

`task_id,domain,understand_plan,execute,debug_verify,explain,difficulty`

Use binary labels (0/1) for the four operations and integers 1/2/3 for
difficulty. Never treat model-generated ratings as independent human ratings.

## Operation decision rules

1. **understand_plan** (1): interpreting the goal and choosing a strategy
   is necessary, not merely reading a ready-made answer.
2. **execute** (1): producing code or locating the cited scientific evidence
   is necessary.
3. **debug_verify** (1): the task meaningfully requires checking correctness,
   difficult edge cases, negation, evidence consistency, or calculations.
   Routine implicit checking alone is not enough.
4. **explain** (1): the required output includes justification or rationale.
   Do not equate the presence of an algorithm with a requirement to explain it.

Operations are **multi-label**, not mutually exclusive.

## Difficulty reference population

Use a declared, consistent reference standard rather than rating only personal
familiarity. For coding, consider an undergraduate with introductory Python
and programming fundamentals. For SciFact, consider an undergraduate with
general scientific literacy but without specialist biomedical training.
Judge the work needed to interpret the supplied prompt and evidence, not
whether the annotator can recall every biomedical fact from memory.

- **1 (easy):** Direct, few-step task with familiar terminology.
- **2 (medium):** Multiple reasoning steps, nontrivial choices, or some
  unfamiliar technical language that the prompt helps interpret.
- **3 (hard):** Complex dependencies, difficult evidence interpretation,
  or extensive background knowledge required despite supplied material.

The two domain reference populations differ; difficulty bins should therefore
not be interpreted as directly psychometrically equivalent.

## Independence and calibration

- Review this handbook together first.
- Practise on a few tasks *outside* the fixed 40-sample set.
- During independent rating, do not consult each other's answers or the
  existing heuristic labels.
- Preserve first-pass labels. Discuss disagreements only after both complete
  sheets have been submitted.
- If AI guidance is used on a sampled task, record its use in a separate
  methods log and do not describe that rating as strictly unaided.
- Record uncertainty in a separate private notes file; do not silently drop
  uncertain tasks or change the frozen schema.
- Do not revise the rubric in response to observed kappa or desired results.

## Agreement and reporting

Run only after two independent sheets are complete:

```powershell
python data/scripts/audit_agreement.py data/audit/annotations_P3.csv data/audit/annotations_second.csv
```

The script checks full sample coverage, duplicate IDs, domain agreement, and
valid labels. It reports observed agreement and Cohen's kappa by operation
and exact/within-one difficulty agreement. Kappa is undefined when both
raters assign the same constant label throughout; report `n/a` honestly.

**Current state:** human agreement results are pending. These instructions
do not constitute completed human validation.

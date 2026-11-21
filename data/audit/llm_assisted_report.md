# Exploratory LLM-assisted audit vs frozen heuristic labels

**Status:** 40/40 sampled tasks rated by GPT-6 (model-only), 20 coding and 20 science.
**Source:** Local `python data/scripts/audit_llm_assisted.py --report` output
shared on 21 November 2026. The remote generated CSV is not yet populated;
this report must be reconciled with the local CSV before reproducing on GitHub.

**NOT an independent human annotation study.**
Agreement here is model-assisted vs heuristic agreement, not human reliability,
validated accuracy, or independent ground truth.

## Provenance

- Model: GPT-6; tasks: 40.
- Review status: model_only; tasks: 40.
- Original human audit: deferred/not conducted.

## Agreement by domain

| Domain | N | Plan | Execute | Debug/verify | Explain | Difficulty exact |
|---|---:|---:|---:|---:|---:|---:|
| coding | 20 | 100.0% | 100.0% | 45.0% | 65.0% | 45.0% |
| science | 20 | 100.0% | 100.0% | 80.0% | 100.0% | 15.0% |

## Interpretation

- Plan and Execute agree in all sampled items, but may be non-discriminative
  because they are commonly assigned to both domains.
- Coding Debug/verify agrees on 9/20 items, Explain on 13/20, and difficulty
  on 9/20.
- Science Debug/verify agrees on 16/20 and difficulty on only 3/20.
- Disagreement flags uncertainty and rubric sensitivity; it does **not**
  establish that the GPT-6 or heuristic labels are correct.
- No independent human ground truth was collected; no human Cohen's kappa
  or semantic-label accuracy is available.
- The 40-task sample is small and was not designed to validate model
  performance independently.

## Reproduction

1. Copy the completed local `data/audit/annotations_llm_assisted.csv` into
   the project repository (do not overwrite the independent human sheet).
2. Run `python data/scripts/audit_llm_assisted.py --status`.
3. Run `python data/scripts/audit_llm_assisted.py --report`.
4. Verify agreement numbers against this report before treating the snapshot
   as reproducible from the checked-in files.

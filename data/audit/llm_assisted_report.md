# Exploratory LLM-assisted audit vs frozen heuristic labels

**NOT an independent human annotation study.** Agreement here is
model-assisted vs heuristic agreement, not human reliability or gold accuracy.

## Provenance

- Model: GPT-6; tasks: 40
- Review status: model_only; tasks: 40

## Agreement by domain

| Domain | N | Plan | Execute | Debug/verify | Explain | Difficulty exact |
|---|---:|---:|---:|---:|---:|---:|
| coding | 20 | 100.0% | 100.0% | 45.0% | 65.0% | 45.0% |
| science | 20 | 100.0% | 100.0% | 80.0% | 100.0% | 15.0% |

## Interpretation

- Differences identify items for expert review; agreement cannot prove label correctness.
- Outputs depend on the model, rubric, prompt, and human review process.
- Do not report these figures as inter-annotator agreement between two humans.
- Independent human validation remains unavailable/pending.

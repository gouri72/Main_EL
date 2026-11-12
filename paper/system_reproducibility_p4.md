# Draft: System, assistance validation, and reproducibility

## Assistance system and hint validation

HumanFirst-Route separates assistance selection from assistance-language generation. At each task step, the policy selects a route and an assistance level from 0 to 4; the reasoning operation is part of the observed state and constrains the permitted level. A frozen Qwen3-0.6B model generates assistance text offline. The resulting bank is indexed by task, operation, and level, and the model is not called inside the learning loop. Transfer tasks are evaluated without hints.

The current bank contains 4,060 entries for 300 base tasks: 4,030 Qwen-generated entries, 9 handwritten entries, and 21 template fallbacks. Coding hints are checked for overlap with solution n-grams that are not already present in the task prompt. Science hints are checked for an asserted gold label, explicit sentence references, and overlap with evidence text at levels where that overlap is disallowed. Separate compliance rules check level-specific constraints such as question form, length, code content, and whether level 4 contains a complete answer. Rejected generations are retried up to three attempts with rejection feedback before a fallback is used.

On the current generation run, 51 of 4,051 generated first-pass candidates were rejected (1.3%); 30 were accepted after retry and 21 required fallback. These are pipeline validation counts, not estimates of validator accuracy. Rule-based overlap and pattern checks can miss paraphrased solution leakage. A 100-item independent annotation sheet has been prepared, but validator accuracy and inter-annotator agreement are not reported until it is completed.

## Reproducibility

Hint-generation provenance is recorded in `assist/cache/generation_env.json`: Qwen/Qwen3-0.6B, prompt version v7, generation tag `hints-v1`, batch size 16, three attempts, temperatures 0.3/0.6/0.9, top-p 0.9, Python 3.13.16, PyTorch 2.14.1+cu130, Transformers 5.22.1, and a Tesla T4 GPU. The response cache, cleaned first-pass outputs, generation statistics, validator rules, task data, and final hint bank are stored in the repository. `python assist/check_bank.py` verifies bank coverage and flags; `python assist/report.py` prints generation summaries.

Experiment configurations are YAML files. The runner records a hash of the configuration in each result CSV and saves the configuration beside the results. Seeds are specified in the configuration; the current `results/smoke_001` run is a smoke test with three seeds, 50 episodes per seed, one policy, and simulator family A. It is not evidence for the paper's multi-policy or held-out-family claims. Package requirements are listed in `requirements.txt` and `requirements-colab.txt`; those files use version ranges rather than a fully pinned environment lock.

## Figures

`analysis/plots.py` produces fading, immediate-success-versus-transfer, and transfer-bar figures from the runner's CSV schema. Confidence intervals are calculated over per-seed summaries. `analysis/compare.py` provides experiment summaries and paired comparisons. The PDFs currently in `paper/figs/` were produced from smoke-scale data and should be replaced with figures from the completed baseline and held-out-simulator runs before submission.

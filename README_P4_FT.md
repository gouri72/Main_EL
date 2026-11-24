# P4: Qwen3-0.6B LoRA comparison

This adds a separate, reproducible base-versus-LoRA experiment. It does not alter the original validated hint bank, Qwen cache, prompts, or project architecture. All generated experiment files live under `results/qwen3_06b_lora_v1/` and should not be confused with the frozen bank.

## Current status

- Dataset preparation and leakage checks have run locally. With seed 42, the manifest groups each base task with its transfer task and groups exact or very-near science abstracts (five-token-shingle Jaccard at least 0.95). It produced 208/46/46 base tasks in train/dev/test (416/92/92 including transfer variants) and 2,822/599/618 retained examples.
- The reproducible split snapshot is checked in at `assist/ft/manifests/qwen3_06b_lora_v1_split.csv` with source hashes/counts in the adjacent JSON. Regenerate the working copy under `results/` with the preparation script.
- The filter retains only base-task Qwen or handwritten hints that pass the current rule validator; the 21 fallback records are excluded. Actual retained targets: 4,030 Qwen-generated pseudo-labels and 9 handwritten hints. The validator is not a correctness or teaching-quality gold standard.
- Both conditions load the same pinned Hugging Face model revision (`c1899de289a04d12100db370d81485cdf75e47ca`) so future model-repository changes cannot silently alter the comparison.
- Training and the paired first-pass evaluation have now completed in Colab. Compact summaries are recorded in [`results/qwen3_06b_lora_v1/RESULTS.md`](results/qwen3_06b_lora_v1/RESULTS.md), `training/training_metadata.json`, and `evaluation/summary.json`. Adapter weights and raw paired outputs remain in Google Drive; no human ratings have been completed.
- Aggregate SVG figures are checked in under `results/qwen3_06b_lora_v1/figures/`. They can be regenerated from the committed summary with `python3 assist/ft/plot_recorded_summary.py`. The full-data chart script below remains available when the original `comparison_metrics.csv` is present.
- The workspace checkout does not contain the Drive adapter, `run_manifest.json`, or raw `paired_outputs.jsonl`. Run `verify_lora_artifacts.py` against the team Drive copy before stating that adapter loading and raw-output access were independently verified. Run `recompute_lora_summary.py` against the same folder to regenerate the summary and all metric/error CSVs without loading a model.
- The existing 120-pair blind sheet has only one set of scores and three dimensions. `prepare_two_rater_sheets.py` creates separate sheets for two independent raters and adds usefulness; the sheets must be completed by actual people before `summarize_human_ratings.py` can report scores and agreement.
- `prepare_error_review_sheets.py` uses the actual paired error CSV plus the prepared test prompts to sample cases from each validator pass/fail category, hide the model identities, and collect semantic error tags and explanations from reviewers.

## Colab runbook (T4-friendly start)

Use a new Colab notebook, select a GPU runtime, and keep the runtime alive until the checkpoint is copied to Drive. Run these cells. If you have the repository cloned in Drive already, set `REPO` to that directory instead of cloning another copy.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content
!git clone https://github.com/gouri72/Main_EL.git
%cd /content/Main_EL
!git switch main
!git pull --ff-only origin main
!git rev-parse HEAD
!nvidia-smi
!python -m pip install -r assist/ft/requirements-ft.txt
!python -m pip uninstall -y torchao
```

Colab may have an older optional `torchao` package preinstalled. It is not needed for this fp16 LoRA run; the training script checks for incompatible versions before downloading the model weights.

Run data preparation and dry runs. Preparation is CPU-only and should not be repeated with different settings after examining test performance. Copy the prepared split to Drive so it survives a Colab reset.

```python
!python assist/ft/prepare_dataset.py
!mkdir -p /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1
!cp -a results/qwen3_06b_lora_v1/prepared /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/
!python assist/ft/train_lora.py --config configs/qwen3_06b_lora_v1.yaml --dry-run
!python assist/ft/evaluate_lora.py --config configs/qwen3_06b_lora_v1.yaml --dry-run
```

Train directly to Drive so periodic checkpoints survive a Colab disconnect. The configured effective batch size is 16 (micro-batch 1 × accumulation 16), with 3 epochs, LR 1e-4, max sequence length 2048 and LoRA rank 8/alpha 16. Checkpoints are saved every 100 optimizer steps, with validation loss at the same interval. T4 memory is limited; if OOM occurs, first change max length to 1536, use a fresh versioned output directory and restart. If that still fails, a separate 4-bit run requires explicit configuration and must have a different experiment ID. To resume, pass the latest checkpoint directory under Drive with `--resume-from-checkpoint`.

```python
!python assist/ft/train_lora.py --config configs/qwen3_06b_lora_v1.yaml --output-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/training
```

If the Colab runtime resets during training, reconnect Drive, clone/pull the same `main` commit, prepare the data with the same seed, and resume from the latest Drive checkpoint (replace the path with the highest numbered checkpoint that exists). Do not start a fresh run in the same directory.

```python
!python assist/ft/train_lora.py --config configs/qwen3_06b_lora_v1.yaml --output-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/training --resume-from-checkpoint /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/training/checkpoints/checkpoint-100
```

If an earlier script version stops with `chat template prefix mismatch`, update the repository before retrying. The current trainer masks prompt tokens directly and supervises only response tokens plus Qwen's end-of-turn token. That failure occurs before optimizer training or checkpoint creation; rerun without `--resume-from-checkpoint`.

The trainer also maps the configured 3% warmup to the appropriate `TrainingArguments` parameter: `warmup_ratio` on older releases or fractional `warmup_steps` on newer Transformers 5 releases.

```python
%cd /content/Main_EL
!git pull --ff-only origin main
```

Run the full paired test. Each test example is generated once by each model with the same prompt, token cap, decoding settings and example seed. Generation alternates model order; timing excludes model loading. Outputs are first-pass and are not regenerated or replaced by fallbacks. Every completed pair is appended to Drive; if Colab resets, rerun the same command and it will continue with unfinished examples after checking the test manifest.

```python
!python assist/ft/evaluate_lora.py --config configs/qwen3_06b_lora_v1.yaml --training-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/training --output-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation
```

If Colab time is short, first make a clearly marked pilot with `--max-examples 64` and a separate output directory; it is not the final result. Do not treat pilot observations as final-test evidence. If evaluation resumes after a reset, use the same full-test command and the same Drive output directory.

Generate the two presentation charts after evaluation:

```python
!python -m pip install matplotlib
!python assist/ft/plot_lora_eval.py --config configs/qwen3_06b_lora_v1.yaml --evaluation-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation
```

The aggregate figures already committed to the repository can be regenerated without installing matplotlib:

```python
!python assist/ft/plot_recorded_summary.py
```

For the requested two-rater protocol, prepare independent forms from the already generated blind sample. Share one CSV with each rater; do not share the key. Raters independently score relevance, factual correctness, level fit, usefulness (each 1–5), and A/B/tie preference. Keep the A/B assignment unchanged across both forms so agreement is calculated on matched outputs. Lock both forms before revealing the key.

```python
!python assist/ft/prepare_two_rater_sheets.py --sheet /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation/blind_annotation_sheet.csv --output-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/ratings_to_share
```

After both independent sheets are completed, summarize scores and agreement:

```python
!python assist/ft/summarize_human_ratings.py --rater1 /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/ratings_to_share/blind_rater_1.csv --rater2 /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/ratings_to_share/blind_rater_2.csv --key /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation/annotation_key.csv --output /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation/human_rating_summary.json
```

Prepare blinded semantic error-review examples after the full evaluation CSV is available. Reviewers should tag answer leakage, incorrect guidance, operation mismatch, assistance-level mismatch, no identified error, or other, and explain their judgments. The automatic pass/fail category remains in the private key.

```python
!python assist/ft/prepare_error_review_sheets.py --cases /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation/paired_error_cases.csv --test-file results/qwen3_06b_lora_v1/prepared/test.jsonl --output-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/error_review --per-category 5
```

Verify the Drive artifacts, check adapter loading with a four-token smoke generation, and regenerate aggregate results from raw outputs without model inference:

```python
!python assist/ft/verify_lora_artifacts.py --config configs/qwen3_06b_lora_v1.yaml --training-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/training --evaluation-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation --load-adapter
!python assist/ft/recompute_lora_summary.py --config configs/qwen3_06b_lora_v1.yaml --evaluation-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation --output-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation/recomputed --compare-summary /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation/summary.json
```

## Outputs and interpretation

- `prepared/{train,dev,test}.jsonl`, `split_manifest.csv`, `summary.json`: versioned data and grouping provenance.
- `training/checkpoints/`, `training/adapter/`, `training_metadata.json`: checkpoints, adapter and recorded run details.
- `evaluation/paired_outputs.jsonl`, `summary.json`, `comparison_metrics.csv`, `validator_failures.csv`: first-pass raw/cleaned outputs, paired rule-validator metrics by domain/level, task-cluster bootstrap interval, latency, and example-level validator errors.
- `evaluation/blind_annotation_sheet.csv` and `annotation_key.csv`: randomized A/B source sample and separate mapping. By default it samples 120 pairs, balanced across domain × assistance level and favoring distinct task groups. `prepare_two_rater_sheets.py` creates separate rater copies with relevance, correctness, level fit, usefulness, and preference fields. Do not claim human evaluation until real raters complete the forms. LLM-as-judge results, if added later, must be labeled as such and should not replace human ground truth.
- `evaluation/paired_error_cases.csv`: all four automated pass/fail pair categories (`both_pass`, base-only fail, LoRA-only fail, both fail) with outputs and a blank reviewer explanation column. `prepare_error_review_sheets.py` creates blinded, sampled forms for actual semantic review.
- `evaluation/figures/`: overall model comparison and level-wise compliance/leakage charts.

Automated validator acceptance is `compliance AND not leakage`; report compliance and leakage separately too. Leakage rate means the rule-based validator flagged output and is not a measured false-negative rate. Task-cluster bootstrap resamples group IDs so operation/level variants from one task group are not counted as independent clusters. The validator does not assess operation relevance, factual correctness, helpfulness, or learner outcomes. No statistical significance should be claimed from the bootstrap interval alone.

For the blind sheets, use the same 1–5 anchors for both outputs: 1 = clearly poor, 3 = mixed/partly useful, 5 = clearly strong. Relevance asks whether the response helps the stated operation; correctness asks whether its claims/code/evidence are valid for the provided task; level fit asks whether it gives only the requested amount of help; usefulness asks for an overall judgment of whether the hint would help with the task. Mark preference as A, B, or tie. The summary reports quadratic weighted kappa for ordinal scores and unweighted kappa for preference. Keep the model key hidden until both sheets are locked. This internal annotation is not a human learning study.

## Reproducibility

The YAML config is the run specification. Keep its exact copy with each result. `summary.json` records data source hashes, split seed, target provenance, and the evaluation records the generation settings in config. The training metadata records actual GPU model and elapsed time. Record the Colab `git rev-parse HEAD`, package versions, runtime type, and any changed settings alongside the Drive copy. Do not edit prompts, validators, or hyperparameters based on the test set.

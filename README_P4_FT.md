# P4: Qwen3-0.6B LoRA comparison

This adds a separate, reproducible base-versus-LoRA experiment. It does not alter the original validated hint bank, Qwen cache, prompts, or project architecture. All generated experiment files live under `results/qwen3_06b_lora_v1/` and should not be confused with the frozen bank.

## Current status

Chapter-ready P4 text, figures and limitations are collected in [`paper/p4_book_chapter_contribution.md`](paper/p4_book_chapter_contribution.md).

- Dataset preparation and leakage checks have run locally. With seed 42, the manifest groups each base task with its transfer task and groups exact or very-near science abstracts (five-token-shingle Jaccard at least 0.95). It produced 208/46/46 base tasks in train/dev/test (416/92/92 including transfer variants) and 2,822/599/618 retained examples.
- The reproducible split snapshot is checked in at `assist/ft/manifests/qwen3_06b_lora_v1_split.csv` with source hashes/counts in the adjacent JSON. Regenerate the working copy under `results/` with the preparation script.
- The filter retains only base-task Qwen or handwritten hints that pass the current rule validator; the 21 fallback records are excluded. Actual retained targets: 4,030 Qwen-generated pseudo-labels and 9 handwritten hints. The validator is not a correctness or teaching-quality gold standard.
- Both conditions load the same pinned Hugging Face model revision (`c1899de289a04d12100db370d81485cdf75e47ca`) so future model-repository changes cannot silently alter the comparison.
- Training and the paired first-pass evaluation have completed in Colab. The raw paired outputs, run manifest, comparison metrics, validator failure list, paired-case table, and training log are checked in with the compact summary. The adapter and checkpoints remain in Google Drive. Independent human ratings were **not conducted**.
- Aggregate SVG figures are checked in under `results/qwen3_06b_lora_v1/figures/`. They can be regenerated from the committed summary with `python3 assist/ft/plot_recorded_summary.py`. The full-data chart script below remains available when the original `comparison_metrics.csv` is present.
- All 618 paired outputs are committed at `results/qwen3_06b_lora_v1/evaluation/paired_outputs.jsonl`; all IDs and prompt hashes match the frozen prepared test split, the rows contain the required paired result fields, and the saved numeric summary reproduces exactly. Adapter-file hashes in `run_manifest.json` were checked against the supplied handoff archive. The adapter and checkpoints remain in Drive. The original Colab evaluation loaded the adapter successfully; a fresh CUDA load/generation smoke test was not performed in this workspace.
- Independent human ratings were **not conducted** and will not be reported. The former blinded-sheet workflow is not part of the final study. Qualitative semantic observations in `ERROR_ANALYSIS.md` are unblinded text inspection, not independent human findings.
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

Verify the Drive artifacts, check adapter loading with a four-token smoke generation, and regenerate aggregate results from raw outputs without model inference:

```python
!python assist/ft/verify_lora_artifacts.py --config configs/qwen3_06b_lora_v1.yaml --training-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/training --evaluation-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation --load-adapter
!python assist/ft/recompute_lora_summary.py --config configs/qwen3_06b_lora_v1.yaml --evaluation-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation --output-dir /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation/recomputed --compare-summary /content/drive/MyDrive/HumanFirst-Route/qwen3_06b_lora_v1/evaluation/summary.json
```

## Outputs and interpretation

- `prepared/{train,dev,test}.jsonl`, `split_manifest.csv`, `summary.json`: versioned data and grouping provenance.
- `training/checkpoints/`, `training/adapter/`, `training_metadata.json`: checkpoints, adapter and recorded run details.
- `evaluation/paired_outputs.jsonl`, `run_manifest.json`, `summary.json`, `comparison_metrics.csv`, `validator_failures.csv`, and `paired_error_cases.csv`: first-pass raw/cleaned outputs, hashed run config and adapter files, rule-validator metrics by domain/level/operation, task-cluster bootstrap interval, latency, and example-level cases.
- `evaluation/blind_annotation_sheet.csv` and `annotation_key.csv` are retained as unused artifacts from an earlier proposed protocol; they are not evidence and no rating exercise was conducted. Do not use them to imply human evaluation.
- `evaluation/paired_error_cases.csv` and [`ERROR_ANALYSIS.md`](results/qwen3_06b_lora_v1/ERROR_ANALYSIS.md): all four automated pass/fail pair categories with actual output text, category counts, and a clearly labeled qualitative inspection. `prepare_error_review_sheets.py` creates blinded, sampled forms for independent human semantic review.
- `evaluation/figures/`: overall model comparison and level-wise compliance/leakage charts.

Automated validator acceptance is `compliance AND not leakage`; report compliance and leakage separately too. Leakage rate means the rule-based validator flagged output and is not a measured false-negative rate. Task-cluster bootstrap resamples group IDs so operation/level variants from one task group are not counted as independent clusters. The validator does not assess operation relevance, factual correctness, helpfulness, or learner outcomes. No statistical significance should be claimed from the bootstrap interval alone.

Human evaluation was not conducted. Validator acceptance is not a proxy for factual correctness, pedagogical usefulness, or learner outcomes.

## Reproducibility

The YAML config is the run specification. Keep its exact copy with each result. `summary.json` records data source hashes, split seed, target provenance, and the evaluation records the generation settings in config. The training metadata records actual GPU model and elapsed time. Record the Colab `git rev-parse HEAD`, package versions, runtime type, and any changed settings alongside the Drive copy. Do not edit prompts, validators, or hyperparameters based on the test set.

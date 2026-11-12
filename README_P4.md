# P4: assistance generation, validator, runner, figures (run from the repo root)

Current-bank checks:

```bash
python tests/test_validator.py
python assist/check_bank.py
python assist/report.py
python assist/eval_validator.py --sheet assist/labels/label_sheet.csv --primary p4 --second p3
```

Experiment and figure commands:

```bash
python run_experiment.py configs/example.yaml
python analysis/plots.py results/<exp_id>
```

`assist/build_stub_hints.py` is only for the initial fallback stub and overwrites `data/hints.jsonl`; do not run it on
the completed Qwen bank. `assist/generate_hints.py --backend qwen --limit 3` is only a small GPU smoke run and writes an
incomplete bank; use the full generation workflow below for a final bank. `assist/make_label_sheet.py` recreates and
overwrites the annotation CSV, so do not rerun it after anyone has entered labels unless you have saved those labels.

In data/hints.jsonl `leakage_flag=True` means the hint leaks the solution/gold label at a level where that is not allowed
(levels 1-3). Level 4 = full assistance, never leak-flagged. `compliance_flag=True` means OK. Hints exist only for base
tasks; transfer variants (*_T) are unaided. Handwritten hints go in data/hints_handwritten.jsonl and override templates.
Validator limitations: rule-based n-gram/regex checks miss paraphrased leaks.

## Current P4 state

The current checked-in bank has 4,060 rows covering all 300 base tasks, their labeled operations, and levels 1-4:
4,030 validated Qwen3-0.6B hints, 9 handwritten hints, and 21 fallback hints. The generation cache contains 4,224
entries; generation details are recorded in `assist/cache/generation_env.json`, and `assist/report.py` summarizes
provenance and first-pass rejection rates. `python assist/check_bank.py` verifies coverage and flags.

The validator sanity script passes. A 100-item hand-label sheet is prepared at `assist/labels/label_sheet.csv` with
30 dev and 70 test items sampled from raw first-pass generations. Validator accuracy remains unmeasured until the
independent labels are filled in; the test split must not be used to tune rules.

The P4 reading note for Bastani et al. is in `paper/notes_p4_bastani.md`. Plotting scripts and PDFs are present, but
the checked-in `results/smoke_001` is one policy in one simulator family, so those plots are pipeline examples only;
they are not the final multi-baseline or held-out-simulator paper figures. `analysis/plots.py` needs the packages in
`requirements.txt`; in the current local Python environment pandas and matplotlib are missing, so I have not
regenerated or overwritten the existing PDFs.

## Colab generation with resumable Drive cache

Use a GPU runtime. Keep `assist/cache` on Google Drive so generated responses survive a runtime reset. The model weights
may need to download again in a new runtime. Run coding and science separately; after each finishes, copy its bank,
raw first-pass generations, and stats to Drive. If a run disconnects, reconnect, remount Drive, restore the cache link,
and rerun the same command; cached generations are reused. Use a small batch size (`4`, or `2` on out-of-memory).

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone https://github.com/gouri72/Main_EL.git
%cd /content/Main_EL
!pip install -r requirements-colab.txt
!nvidia-smi
```

```python
from pathlib import Path
repo = Path("/content/Main_EL")
drive_cache = Path("/content/drive/MyDrive/HumanFirstRoute/assist-cache")
drive_cache.mkdir(parents=True, exist_ok=True)
cache_link = repo / "assist" / "cache"
if cache_link.is_symlink():
    cache_link.unlink()
elif cache_link.exists():
    raise RuntimeError(f"Inspect existing cache path before replacing: {cache_link}")
cache_link.symlink_to(drive_cache, target_is_directory=True)
```

```bash
!mkdir -p /content/drive/MyDrive/HumanFirstRoute/generated
!python assist/generate_hints.py --backend qwen --domain coding --batch-size 4 --retries 3 --seed 0
!cp data/hints.jsonl /content/drive/MyDrive/HumanFirstRoute/generated/hints_coding.jsonl
!cp assist/cache/raw_first_pass.jsonl /content/drive/MyDrive/HumanFirstRoute/generated/raw_coding.jsonl
!cp assist/cache/stats.json /content/drive/MyDrive/HumanFirstRoute/generated/stats_coding.json
```

```bash
!python assist/generate_hints.py --backend qwen --domain science --batch-size 4 --retries 3 --seed 0
!cp data/hints.jsonl /content/drive/MyDrive/HumanFirstRoute/generated/hints_science.jsonl
!cp assist/cache/raw_first_pass.jsonl /content/drive/MyDrive/HumanFirstRoute/generated/raw_science.jsonl
!cp assist/cache/stats.json /content/drive/MyDrive/HumanFirstRoute/generated/stats_science.json
```

Copy the six saved files from Drive into the repo, then merge and validate complete coverage:

```bash
!cp /content/drive/MyDrive/HumanFirstRoute/generated/hints_coding.jsonl data/hints_coding.jsonl
!cp /content/drive/MyDrive/HumanFirstRoute/generated/hints_science.jsonl data/hints_science.jsonl
!cp /content/drive/MyDrive/HumanFirstRoute/generated/raw_coding.jsonl assist/cache/raw_coding.jsonl
!cp /content/drive/MyDrive/HumanFirstRoute/generated/raw_science.jsonl assist/cache/raw_science.jsonl
!cp /content/drive/MyDrive/HumanFirstRoute/generated/stats_coding.json assist/cache/stats_coding.json
!cp /content/drive/MyDrive/HumanFirstRoute/generated/stats_science.json assist/cache/stats_science.json
```

```bash
python assist/merge_hint_runs.py \
  --hints coding=data/hints_coding.jsonl --hints science=data/hints_science.jsonl \
  --raw coding=assist/cache/raw_coding.jsonl --raw science=assist/cache/raw_science.jsonl \
  --stats coding=assist/cache/stats_coding.json --stats science=assist/cache/stats_science.json
```

The merger refuses incomplete or duplicate domain banks and writes a combined bank, raw first-pass file, and aggregate
stats. Then create and fill the independent annotation sheet:

```bash
python assist/make_label_sheet.py --n 100 --seed 0
python assist/eval_validator.py --sheet assist/labels/label_sheet.csv --primary p4 --second p3
```

The accuracy report is not complete until labels are filled. The `p3` columns are optional for accuracy, but required
to report inter-annotator agreement. Save the completed generated files and annotation sheet in the repo so the run is
reproducible; do not describe fallback rows as Qwen output.

## Hint bank provenance
- Qwen3-0.6B outputs (assist/cache/hints_cache.jsonl, 4,224 entries) were generated at git tag hints-v1 (prompt v7); environment in assist/cache/generation_env.json.
- data/hints.jsonl is assembled from that cache with the validator rules in this commit (assist/rules.yaml): `python assist/generate_hints.py --backend cacheonly`. Level-2 min_words was loosened from 10 to 6 after generation (a loosening needs no new generation); nothing else changed.
- `python assist/check_bank.py` verifies coverage; `python assist/report.py` prints the summary numbers.

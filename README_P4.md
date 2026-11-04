# P4: assistance generation, validator, runner, figures (run from the repo root)

    pip install -r requirements.txt
    python tests/test_validator.py                         # validator sanity script (must print ALL PASS)
    python assist/build_stub_hints.py                      # Day-1 stub: full ladder for all base tasks -> data/hints.jsonl
    python assist/generate_hints.py --backend mock         # offline test of generate/validate/retry/fallback
    python assist/generate_hints.py --backend qwen --limit 3   # Colab GPU, de-risk run (pip install -r requirements-colab.txt)
    python run_experiment.py configs/example.yaml          # -> results/<exp_id>/{steps,episodes,config}.csv/yaml
    python analysis/plots.py results/<exp_id>              # -> paper/figs/*.pdf
    python assist/make_label_sheet.py --n 100              # after the real Qwen run: hand-labeling sheet
    python assist/eval_validator.py                        # validator precision/recall + inter-annotator kappa

In data/hints.jsonl `leakage_flag=True` means the hint leaks the solution/gold label at a level where that is not allowed
(levels 1-3). Level 4 = full assistance, never leak-flagged. `compliance_flag=True` means OK. Hints exist only for base
tasks; transfer variants (*_T) are unaided. Handwritten hints go in data/hints_handwritten.jsonl and override templates.
Validator limitations: rule-based n-gram/regex checks miss paraphrased leaks.

## Current P4 state

The checked-in `data/hints.jsonl` is a usable stub bank, not the final Qwen bank. In the inspected snapshot it had
4,044 rows: 9 handwritten and 4,035 fallback, with no generation cache or validator label sheet. The validator sanity
script passed, but validator accuracy has not been measured until hints are independently hand-labeled.

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

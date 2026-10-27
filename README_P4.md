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
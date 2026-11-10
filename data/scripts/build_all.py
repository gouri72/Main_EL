#!/usr/bin/env python3
"""Rebuild the whole task set from data/build_config.yaml:
download -> coding -> relabel coding ops -> science -> repair science variants -> merge+validate
-> hint stubs -> stats -> data checks.

  python data/scripts/build_all.py             # run everything
  python data/scripts/build_all.py --dry-run   # only print the commands

WARNING: the hint-stub step rewrites data/hints.jsonl (fallback hints). Merge real Qwen hints
afterwards (assist/merge_hint_runs.py) or pass --skip-hints to leave data/hints.jsonl alone.
"""
import argparse, subprocess, sys
from pathlib import Path

import yaml

D = Path(__file__).resolve().parents[1]
S = D / "scripts"
ASSIST = D.parent / "assist"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-hints", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((D / "build_config.yaml").read_text(encoding="utf-8"))
    py, seed = sys.executable, str(cfg["seed"])
    steps = [
        [py, str(S / "download_data.py")],
        [py, str(S / "build_coding.py"), "--n", str(cfg["coding_tasks"]), "--seed", seed],
        [py, str(S / "relabel_coding_ops.py")],
        [py, str(S / "build_science.py"), "--n", str(cfg["science_tasks"]), "--seed", seed],
        [py, str(S / "repair_science_variants.py")],
        [py, str(S / "merge_tasks.py")],
    ]
    if not a.skip_hints:
        steps.append([py, str(ASSIST / "build_stub_hints.py")])
    steps += [
        [py, str(S / "dataset_stats.py")],
        [py, str(S / "data_checks.py")],
    ]
    for s in steps:
        print(">>", Path(s[1]).name, " ".join(s[2:]))
        if not a.dry_run and subprocess.call(s) != 0:
            sys.exit("step failed: " + Path(s[1]).name)


if __name__ == "__main__":
    main()
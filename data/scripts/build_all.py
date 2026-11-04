#!/usr/bin/env python3
"""Rebuild the whole task set from data/build_config.yaml:
download -> coding -> science -> repair science variants -> merge+validate -> stats -> data checks.

  python data/scripts/build_all.py             # run everything
  python data/scripts/build_all.py --dry-run   # only print the commands
"""
import argparse, subprocess, sys
from pathlib import Path

import yaml

D = Path(__file__).resolve().parents[1]
S = D / "scripts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((D / "build_config.yaml").read_text(encoding="utf-8"))
    py, seed = sys.executable, str(cfg["seed"])
    steps = [
        [py, str(S / "download_data.py")],
        [py, str(S / "build_coding.py"), "--n", str(cfg["coding_tasks"]), "--seed", seed],
        [py, str(S / "build_science.py"), "--n", str(cfg["science_tasks"]), "--seed", seed],
        [py, str(S / "repair_science_variants.py")],
        [py, str(S / "merge_tasks.py")],
        [py, str(S / "dataset_stats.py")],
        [py, str(S / "data_checks.py")],
    ]
    for s in steps:
        print(">>", Path(s[1]).name, " ".join(s[2:]))
        if not a.dry_run and subprocess.call(s) != 0:
            sys.exit(f"step failed: {Path(s[1]).name}")


if __name__ == "__main__":
    main()
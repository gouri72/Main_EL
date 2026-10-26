#!/usr/bin/env python3
"""Download raw datasets into data/raw/ (stdlib only, no pip install needed).

  python data/scripts/download_data.py
"""
import tarfile, urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "raw"
RAW.mkdir(parents=True, exist_ok=True)

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already have {dest.name}")
        return True
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


fetch(MBPP_URL, RAW / "mbpp.jsonl")

tgz = RAW / "scifact_data.tar.gz"
if fetch(SCIFACT_URL, tgz):
    out = RAW / "scifact"
    if not out.exists():
        with tarfile.open(tgz) as t:
            t.extractall(out)
    print("SciFact files found:", sorted(p.name for p in out.rglob("*.jsonl")))
else:
    print("\nSciFact download failed. Open https://github.com/allenai/scifact , follow the README "
          "download link, save the file as data/raw/scifact_data.tar.gz and run this script again.")

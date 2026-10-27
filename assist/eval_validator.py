#!/usr/bin/env python3
"""Validator accuracy vs hand labels (+ inter-annotator kappa). Report TEST split only in the paper.
  python assist/eval_validator.py --sheet assist/labels/label_sheet.csv --primary p4 --second p3"""
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions

def kappa(a, b):
    n = len(a)
    if n == 0: return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == v for x in a) / n) * (sum(y == v for y in b) / n) for v in (0, 1))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def prf(pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold)); fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    return (tp / (tp + fp) if tp + fp else float("nan"), tp / (tp + fn) if tp + fn else float("nan"))

ap = argparse.ArgumentParser()
ap.add_argument("--sheet", default="assist/labels/label_sheet.csv")
ap.add_argument("--primary", default="p4"); ap.add_argument("--second", default="p3")
a = ap.parse_args()
T, S = load_tasks(), load_solutions()
rows = list(csv.DictReader(open(a.sheet, encoding="utf-8")))
for split in ("dev", "test"):
    rs = [r for r in rows if r["split"] == split and r[f"{a.primary}_leak"] != ""]
    if not rs: print(split, "no labels yet"); continue
    pl, pc, gl, gc = [], [], [], []
    for r in rs:
        v = validate(T[r["task_id"]], r["op"], int(r["level"]), r["text"], S.get(r["task_id"]))
        pl.append(v["leakage_flag"]); pc.append(v["compliance_flag"])
        gl.append(r[f"{a.primary}_leak"] == "1"); gc.append(r[f"{a.primary}_compliant"] == "1")
    acc_l = sum(p == g for p, g in zip(pl, gl)) / len(rs); acc_c = sum(p == g for p, g in zip(pc, gc)) / len(rs)
    pr, rc = prf(pl, gl)
    print(f"[{split}] n={len(rs)} leakage acc={acc_l:.2f} precision={pr:.2f} recall={rc:.2f} | compliance acc={acc_c:.2f}")
    s2 = [r for r in rs if r.get(f"{a.second}_leak", "") != ""]
    if s2:
        k1 = kappa([int(r[f"{a.primary}_leak"]) for r in s2], [int(r[f"{a.second}_leak"]) for r in s2])
        k2 = kappa([int(r[f"{a.primary}_compliant"]) for r in s2], [int(r[f"{a.second}_compliant"]) for r in s2])
        print(f"[{split}] inter-annotator kappa leak={k1:.2f} compliant={k2:.2f} (n={len(s2)})")
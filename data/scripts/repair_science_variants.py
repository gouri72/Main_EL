#!/usr/bin/env python3
"""Re-pick the science transfer variants so that no variant reuses an abstract that any session task uses,
and no two variants share an abstract. Session tasks (ids, prompts, gold) are NOT touched, so P4's hints and
everyone's sessions stay valid. Variants keep the same label, evidence-size bucket and difficulty as their base;
among the allowed candidates the most similar claim (TF-IDF cosine over claim + title words) is chosen.

  python data/scripts/repair_science_variants.py
  python data/scripts/merge_tasks.py
Needs data/raw/scifact (download_data.py).
"""
import json, math, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_science as bs  # noqa: E402

D = Path(__file__).resolve().parents[1]
P = D / "tasks_science.jsonl"


def claim_of(row):
    m = re.match(r"Claim: (.*?)\n\nAbstract \(", row["prompt"], re.S)
    return m.group(1).strip() if m else None


def main():
    rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]
    cands = bs.build_candidates()
    by_claim = {c["claim"]: c for c in cands}
    df = Counter(w for c in cands for w in c["words"])
    n = len(cands)
    idf = {w: math.log((n + 1) / (df[w] + 1)) + 1 for w in df}

    def cos(a, b):
        num = sum(idf[w] ** 2 for w in a & b)
        den = math.sqrt(sum(idf[w] ** 2 for w in a)) * math.sqrt(sum(idf[w] ** 2 for w in b))
        return num / den if den else 0.0

    base = [r for r in rows if not r["task_id"].endswith("_T")]
    prim = {}
    for r in base:
        c = by_claim.get(claim_of(r))
        if c is None:
            sys.exit(f"{r['task_id']}: claim not found in SciFact candidates; was tasks_science.jsonl built from this data?")
        if c["label"] != r["gold"]["label"]:
            sys.exit(f"{r['task_id']}: label mismatch with SciFact; aborting")
        prim[r["task_id"]] = c
    banned_cids = {c["cid"] for c in prim.values()}
    banned_docs = {c["doc_id"] for c in prim.values()}
    pool = [c for c in cands if c["cid"] not in banned_cids and c["doc_id"] not in banned_docs]

    used_docs, new, kept = set(), {}, 0
    for r in base:
        c = prim[r["task_id"]]
        opts = [p for p in pool if p["label"] == c["label"] and p["bucket"] == c["bucket"]
                and p["doc_id"] not in used_docs]
        if not opts:
            print(f"WARNING {r['task_id']}: no clean variant available, keeping the old one")
            kept += 1
            continue
        best = max(opts, key=lambda p: cos(c["words"], p["words"]))
        used_docs.add(best["doc_id"])
        new[r["task_id"] + "_T"] = best

    out = []
    diff = {r["task_id"]: r["difficulty"] for r in base}
    for r in rows:
        tid = r["task_id"]
        if tid in new:
            v = new[tid]
            r = {"task_id": tid, "domain": "science", "prompt": bs.prompt_for(v),
                 "difficulty": diff[tid[:-2]], "operations": bs.operations_for(v),
                 "gold": {"label": v["label"], "evidence": v["evidence"]},
                 "transfer_task_id": tid, "error_types": bs.error_types_for(v)}
        out.append(r)
    with open(P, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"re-picked {len(new)} science variants, kept {kept} old; session tasks untouched")
    print("next: python data/scripts/merge_tasks.py ; python data/scripts/data_checks.py")


if __name__ == "__main__":
    main()
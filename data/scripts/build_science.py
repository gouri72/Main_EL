#!/usr/bin/env python3
"""Build SciFact science tasks + matched transfer variants.

  python data/scripts/build_science.py --n 5   --seed 42
  python data/scripts/build_science.py --n 150 --seed 42

Needs data/raw/scifact/ (run download_data.py first). Writes data/tasks_science.jsonl.

Gold = SciFact's annotated label + rationale sentences of ONE cited abstract (claims whose evidence has
mixed labels are skipped). The prompt contains the claim and that abstract.
Transfer variant = a DIFFERENT claim/abstract with the same label and same evidence-size bucket (1, 2, 3+
sentences), chosen to be the most lexically similar (Jaccard) among unused claims.
Difficulty (tertiles of evidence size + abstract length) and error types are HEURISTIC.
Operation labels: 'execute' (find evidence) is gold-grounded; the rest are heuristic.
"""
import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "scifact"
OUT = ROOT / "data" / "tasks_science.jsonl"


def find(name):
    hits = sorted(RAW.rglob(name))
    return hits[0] if hits else None


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def words(s):
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def build_candidates():
    corpus_p = find("corpus.jsonl")
    if not corpus_p:
        sys.exit("corpus.jsonl not found under data/raw/scifact. Run download_data.py (or check the extraction).")
    corpus = {int(d["doc_id"]): d for d in load_jsonl(corpus_p)}
    claims = []
    for nm in ("claims_train.jsonl", "claims_dev.jsonl"):
        p = find(nm)
        if p:
            claims += load_jsonl(p)
    if not claims:
        sys.exit("No claims_train/dev.jsonl found under data/raw/scifact.")
    cands = []
    for c in claims:
        ev = c.get("evidence") or {}
        if not ev:
            continue
        labels = {e["label"] for sets in ev.values() for e in sets}
        if len(labels) != 1:
            continue
        label = labels.pop()
        doc_id = sorted(ev.keys(), key=int)[0]
        doc = corpus.get(int(doc_id))
        if not doc:
            continue
        abstract = doc["abstract"]
        idx = sorted({i for e in ev[doc_id] for i in e["sentences"] if i < len(abstract)})
        evidence = [abstract[i].strip() for i in idx]
        if not evidence:
            continue
        cands.append({"cid": c["id"], "claim": c["claim"].strip(), "label": label, "doc_id": int(doc_id),
                      "title": doc.get("title", "").strip(), "abstract": [s.strip() for s in abstract],
                      "evidence": evidence, "bucket": min(len(evidence), 3),
                      "words": words(c["claim"] + " " + doc.get("title", ""))})
    return cands


def prompt_for(c):
    sents = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(c["abstract"]))
    return (f"Claim: {c['claim']}\n\nAbstract ({c['title']}):\n{sents}\n\n"
            "Decide whether the abstract SUPPORTS or CONTRADICTS the claim, and name the sentence(s) "
            "that justify your decision.")


def error_types_for(c):
    et = []
    if c["label"] == "CONTRADICT":
        et.append("missed_negation")
    if len(c["evidence"]) > 1:
        et.append("multi_sentence_evidence")
    if re.search(r"\d", c["claim"]):
        et.append("numeric_misread")
    return et or ["overgeneralization"]


def operations_for(c):
    ops = [("understand_plan", "heuristic"), ("execute", "gold")]
    if c["label"] == "CONTRADICT" or len(c["evidence"]) > 1:
        ops.append(("debug_verify", "heuristic"))
    ops.append(("explain", "heuristic"))
    return [{"op": o, "label_source": s} for o, s in ops]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cands = build_candidates()
    print(f"usable SciFact claims: {len(cands)} | labels {dict(Counter(c['label'] for c in cands))}")
    rng = random.Random(a.seed)
    rng.shuffle(cands)
    used = set()
    want = {"SUPPORT": (a.n + 1) // 2, "CONTRADICT": a.n // 2}
    got = Counter()
    pairs = []
    for c in cands:
        if sum(got.values()) >= a.n:
            break
        if c["cid"] in used or got[c["label"]] >= want[c["label"]]:
            continue
        pool = [p for p in cands if p["cid"] not in used and p["cid"] != c["cid"]
                and p["label"] == c["label"] and p["bucket"] == c["bucket"] and p["doc_id"] != c["doc_id"]]
        if not pool:
            continue
        partner = max(pool, key=lambda p: jaccard(c["words"], p["words"]))
        used |= {c["cid"], partner["cid"]}
        got[c["label"]] += 1
        pairs.append((c, partner))

    def score(c):
        return len(c["evidence"]) + len(c["abstract"]) / 10
    order = sorted(range(len(pairs)), key=lambda i: score(pairs[i][0]))
    diff = {}
    for rank, i in enumerate(order):
        diff[i] = 1 + min(2, rank * 3 // max(1, len(pairs)))

    rows = []
    for k, (c, v) in enumerate(pairs):
        tid = f"sci_{k+1:03d}"
        for task_id, x in ((tid, c), (tid + "_T", v)):
            rows.append({"task_id": task_id, "domain": "science", "prompt": prompt_for(x),
                         "difficulty": diff[k], "operations": operations_for(x),
                         "gold": {"label": x["label"], "evidence": x["evidence"]},
                         "transfer_task_id": tid + "_T", "error_types": error_types_for(x)})
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} tasks (+{len(pairs)} variants) -> {OUT.name}")
    print("labels:", dict(got), "| difficulty:", dict(sorted(Counter(diff.values()).items())))
    if len(pairs) < a.n:
        print(f"WARNING: only {len(pairs)} of {a.n} requested could be paired")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Draw the label-noise audit sample (default 40 tasks, half coding / half science, balanced over difficulty).

  python data/scripts/audit_sample.py --n 40 --seed 7

Writes (in data/audit/):
  audit_items.txt            full text of each sampled task, for the annotators to read (heuristic labels NOT shown)
  annotations_template.csv   blank sheet; each annotator copies it to annotations_<name>.csv and fills it in
  sample_ids.json            the sampled task ids
Annotators work independently and must not look at data/tasks.jsonl operations/difficulty while labelling.
"""
import argparse, csv, json, random
from pathlib import Path

D = Path(__file__).resolve().parents[1]
OUT = D / "audit"
OUT.mkdir(exist_ok=True)

RUBRIC = """RUBRIC (decide for each task, independently, WITHOUT looking at tasks.jsonl labels)
Put 1 if a learner solving the task from scratch genuinely needs this reasoning operation, else 0.
  understand_plan : must work out what is being asked / choose an approach before acting
                    (coding: interpret the spec, pick an algorithm; science: identify what the claim asserts and what evidence is needed)
  execute         : must carry out the work (coding: write the code; science: locate the relevant evidence sentences)
  debug_verify    : must check the work (coding: edge cases / test the code; science: watch for negation, hedging,
                    numbers, or evidence spread over several sentences)
  explain         : must justify the answer in words (coding: why the approach works; science: why the evidence implies the label)
difficulty: 1 = easy, 2 = medium, 3 = hard, relative to a typical undergraduate.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    tasks = []
    with open(D / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if not t["task_id"].endswith("_T"):
                    tasks.append(t)
    domains = sorted({t["domain"] for t in tasks})
    per = {d: a.n // len(domains) for d in domains}
    for d in domains[: a.n - sum(per.values())]:
        per[d] += 1

    chosen = []
    for dom in domains:
        groups = {k: [t for t in tasks if t["domain"] == dom and t["difficulty"] == k] for k in (1, 2, 3)}
        for g in groups.values():
            rng.shuffle(g)
        picked = []
        while len(picked) < per[dom] and any(groups.values()):
            for k in (1, 2, 3):
                if groups[k] and len(picked) < per[dom]:
                    picked.append(groups[k].pop())
        chosen += picked
    rng.shuffle(chosen)

    with open(OUT / "audit_items.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(RUBRIC + "\n")
        for t in chosen:
            f.write("\n" + "=" * 78 + f"\n{t['task_id']}  [{t['domain']}]\n" + "=" * 78 + "\n")
            f.write(t["prompt"] + "\n")
            g = t["gold"]
            if t["domain"] == "coding":
                f.write("\n--- tests ---\n" + g["tests"] + "\n")
            else:
                f.write(f"\n--- gold label: {g['label']} ---\n")
                for e in g["evidence"]:
                    f.write(f"  * {e}\n")
    with open(OUT / "annotations_template.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "domain", "understand_plan", "execute", "debug_verify", "explain", "difficulty"])
        for t in chosen:
            w.writerow([t["task_id"], t["domain"], "", "", "", "", ""])
    (OUT / "sample_ids.json").write_text(json.dumps([t["task_id"] for t in chosen], indent=1), encoding="utf-8")
    print(f"sampled {len(chosen)} tasks: {dict((d, sum(t['domain']==d for t in chosen)) for d in domains)}")
    print("wrote data/audit/audit_items.txt, annotations_template.csv, sample_ids.json")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Generate the hint bank with Qwen3-0.6B (run on Colab GPU), validate, regenerate up to N times, fall back.
Qwen NEVER runs inside the RL loop; outputs are cached in assist/cache/hints_cache.jsonl (commit it!).

  python assist/generate_hints.py --backend mock                 # offline test of the whole pipeline
  python assist/generate_hints.py --backend qwen --limit 3       # Day-4 de-risk run on 3 tasks
  python assist/generate_hints.py --backend qwen                 # full run (Week 2)

Handwritten hints in data/hints_handwritten.jsonl always win. Writes data/hints.jsonl and assist/cache/stats.json."""
import argparse, hashlib, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions
from assist.fallback import fallback_text
from assist.prompts import PROMPT_VERSION, SYSTEM, build_user_prompt

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "assist" / "cache" / "hints_cache.jsonl"
MODEL_ID = "Qwen/Qwen3-0.6B"   # verify on the Hugging Face page before the full run

PREFIX = re.compile(r"^\s*(?:\*\*)?(?:hint|answer|level\s*[1-4][^:\n]*)\s*:\s*(?:\*\*)?\s*", re.I)

def clean(t):
    """Strip <think> traces and a leading 'Hint:' / 'Level 2:' label before validating."""
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S | re.I).strip()
    t = PREFIX.sub("", t, count=1).strip()
    t = re.sub(r"\s*\?(\s*\?)+", "?", t)                         # "properly? ?" -> "properly?"
    m = re.search(r"```.*?```", t, flags=re.S)                     # drop code-like junk after the closing fence
    if m and re.search(r"\bdef\b|\bfor\b.*:|\breturn\b|[=\[\]{}]", t[m.end():]):
        t = t[:m.end()]
    return t.strip()

class MockBackend:
    """Deterministic fake model that sometimes leaks, so the validator/fallback path gets exercised."""
    def generate(self, system, user, seed, max_new_tokens=160, temperature=0.3):
        return "Think about what the task is asking and what a correct answer must satisfy?" if seed % 2 == 0 else \
               "The answer is that the abstract supports the claim. ```python\nreturn x\n```"

class QwenBackend:
    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    def generate(self, system, user, seed, max_new_tokens=160, temperature=0.3):
        self.torch.manual_seed(seed)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)   # no <think> traces
        ids = self.tok([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=max(temperature, 0.01), top_p=0.9)
        return self.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

def tok_limit(lv, domain):
    """Max new tokens per level: short hints must not ramble, level 4 may carry code or quoted evidence."""
    return {1: 60, 2: 100, 3: 180, 4: 400 if domain == "coding" else 250}[lv]

def key(model, tid, op, lv, attempt):
    return hashlib.sha256(f"{model}|{PROMPT_VERSION}|{tid}|{op}|{lv}|{attempt}".encode()).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "qwen"], default="mock")
    ap.add_argument("--limit", type=int, default=0, help="only the first N base tasks")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", choices=["coding", "science"], default=None, help="only tasks of this domain")
    a = ap.parse_args()
    T, S = load_tasks(), load_solutions()
    backend = MockBackend() if a.backend == "mock" else QwenBackend()
    model = "mock" if a.backend == "mock" else MODEL_ID
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE.exists():
        for l in open(CACHE, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); cache[d["key"]] = d["text"]
    hand = {}
    hp = ROOT / "data" / "hints_handwritten.jsonl"
    if hp.exists():
        for l in open(hp, encoding="utf-8"):
            if l.strip():
                h = json.loads(l); hand[(h["task_id"], h["op"], h["level"])] = h["text"]

    base = [t for tid, t in T.items() if not tid.endswith("_T")]
    if a.domain: base = [t for t in base if t["domain"] == a.domain]
    if a.limit: base = base[:a.limit]
    raw_first = []   # attempt-0 raw model outputs: THIS is what validator accuracy must be measured on
    rows, st = [], {"total": 0, "first_pass_violations": 0, "regenerated_ok": 0, "fallback": 0, "handwritten": 0,
                    "violations_by_level": {1: 0, 2: 0, 3: 0, 4: 0}, "model": model, "prompt_version": PROMPT_VERSION}
    t0 = time.time()
    with open(CACHE, "a", encoding="utf-8") as cf:
        for ti, t in enumerate(base):
            print(f"[{ti+1}/{len(base)}] {t['task_id']}  elapsed {time.time()-t0:.0f}s  first_pass_violations {st['first_pass_violations']}", flush=True)
            sol = S.get(t["task_id"])
            for o in t["operations"]:
                op = o["op"]
                for lv in (1, 2, 3, 4):
                    st["total"] += 1
                    k0 = (t["task_id"], op, lv)
                    if k0 in hand:
                        text, src = hand[k0], "handwritten"; st["handwritten"] += 1
                        v = validate(t, op, lv, text, sol)
                    else:
                        text, src, v = None, "qwen", None
                        for attempt in range(a.retries):
                            kk = key(model, *k0, attempt)
                            if kk not in cache:
                                cache[kk] = backend.generate(SYSTEM, build_user_prompt(t, op, lv, sol), a.seed + attempt, tok_limit(lv, t["domain"]), 0.3 + 0.3 * attempt)
                                cf.write(json.dumps({"key": kk, "text": cache[kk]}, ensure_ascii=False) + "\n"); cf.flush()
                            cand = clean(cache[kk])
                            if attempt == 0:
                                raw_first.append({"task_id": t["task_id"], "op": op, "level": lv, "text": cand})
                            vc = validate(t, op, lv, cand, sol)
                            ok = not vc["leakage_flag"] and vc["compliance_flag"]
                            if attempt == 0 and not ok:
                                st["first_pass_violations"] += 1; st["violations_by_level"][lv] += 1
                            if ok:
                                text, v = cand, vc
                                if attempt > 0: st["regenerated_ok"] += 1
                                break
                        if text is None:   # all retries violated -> safe fallback
                            text, src = fallback_text(t, op, lv, sol), "fallback"; st["fallback"] += 1
                            v = validate(t, op, lv, text, sol)
                    rows.append({"task_id": t["task_id"], "op": op, "level": lv, "text": text,
                                 "leakage_flag": v["leakage_flag"], "compliance_flag": v["compliance_flag"], "source": src})
    with open(ROOT / "data" / "hints.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(CACHE.parent / "raw_first_pass.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in raw_first: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (CACHE.parent / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))

if __name__ == "__main__":
    main()
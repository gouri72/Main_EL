#!/usr/bin/env python3
"""Generate the hint bank with Qwen3-0.6B (Colab GPU), validate, regenerate failures, fall back to templates.
Qwen NEVER runs inside the RL loop; outputs are cached in assist/cache/hints_cache.jsonl (commit it!).

  python assist/generate_hints.py --backend mock                      # offline test of the whole pipeline
  python assist/generate_hints.py --backend qwen --batch-size 16      # real run (GPU); resumes from the cache
  python assist/generate_hints.py --backend cacheonly                 # NO model/GPU: rebuild hints.jsonl + stats from the cache

Generation is batched: for each attempt, every item that still needs a valid hint is generated in GPU batches
(grouped by token limit, sorted by prompt length). Attempt k+1 only runs for items rejected at attempt k.
Handwritten hints in data/hints_handwritten.jsonl always win. Writes data/hints.jsonl, assist/cache/stats.json,
assist/cache/raw_first_pass.jsonl (cleaned attempt-0 outputs: what validator accuracy must be measured on)."""
import argparse, hashlib, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions
from assist.fallback import fallback_text
from assist.prompts import PROMPT_VERSION, SYSTEM, build_user_prompt, feedback_text

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
    def generate_batch(self, system, users, seed, max_new_tokens, temperature):
        return [self.generate(system, u, seed, max_new_tokens, temperature) for u in users]

class QwenBackend:
    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"                       # required for batched generation
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    def generate_batch(self, system, users, seed, max_new_tokens, temperature):
        self.torch.manual_seed(seed)
        texts = [self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for u in users]                            # enable_thinking=False: no <think> traces
        ids = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=max(temperature, 0.01), top_p=0.9, pad_token_id=self.tok.pad_token_id)
        n = ids.input_ids.shape[1]
        return [self.tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]

def tok_limit(lv, domain):
    """Max new tokens per level: short hints must not ramble, level 4 may carry code or quoted evidence."""
    return {1: 60, 2: 100, 3: 180, 4: 400 if domain == "coding" else 250}[lv]

def key(model, tid, op, lv, attempt):
    return hashlib.sha256(f"{model}|{PROMPT_VERSION}|{tid}|{op}|{lv}|{attempt}".encode()).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "qwen", "cacheonly"], default="mock")
    ap.add_argument("--limit", type=int, default=0, help="only the first N base tasks")
    ap.add_argument("--domain", choices=["coding", "science"], default=None, help="only tasks of this domain")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16, help="lower to 8 if the GPU runs out of memory")
    ap.add_argument("--shard", default=None, help="K/N: only base tasks with index %% N == K; writes only the cache")
    ap.add_argument("--dry-run", action="store_true", help="print how many items this run covers, then exit (no model)")
    a = ap.parse_args()
    T, S = load_tasks(), load_solutions()
    model = "mock" if a.backend == "mock" else MODEL_ID
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE.exists():
        for l in open(CACHE, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); cache[d["key"]] = d["text"]
    print(f"cache entries loaded: {len(cache)}  ({CACHE})", flush=True)
    hand = {}
    hp = ROOT / "data" / "hints_handwritten.jsonl"
    if hp.exists():
        for l in open(hp, encoding="utf-8"):
            if l.strip():
                h = json.loads(l); hand[(h["task_id"], h["op"], h["level"])] = h["text"]

    base = [t for tid, t in T.items() if not tid.endswith("_T")]
    if a.shard:                                   # shard on the FULL base list so every person gets identical shards
        k, n = (int(x) for x in a.shard.split("/"))
        assert 0 <= k < n, "--shard must look like K/N with 0 <= K < N"
        base = [t for j, t in enumerate(base) if j % n == k]
    if a.domain: base = [t for t in base if t["domain"] == a.domain]
    if a.limit: base = base[:a.limit]
    items = [(t, o["op"], lv) for t in base for o in t["operations"] for lv in (1, 2, 3, 4)]   # fixed order
    st = {"total": len(items), "first_pass_violations": 0, "regenerated_ok": 0, "fallback": 0, "handwritten": 0,
          "missing_cache": 0, "violations_by_level": {1: 0, 2: 0, 3: 0, 4: 0}, "model": model,
          "prompt_version": PROMPT_VERSION}
    if a.dry_run:
        nh = sum((t["task_id"], op, lv) in hand for t, op, lv in items)
        print(f"dry run: {len(base)} tasks, {len(items)} items ({nh} handwritten, {len(items) - nh} to generate)")
        return
    backend = {"mock": MockBackend, "qwen": QwenBackend, "cacheonly": lambda: None}[a.backend]()
    resolved, raw_first = {}, {}
    for i, (t, op, lv) in enumerate(items):
        if (t["task_id"], op, lv) in hand:
            text = hand[(t["task_id"], op, lv)]
            resolved[i] = (text, "handwritten", validate(t, op, lv, text, S.get(t["task_id"]))); st["handwritten"] += 1
    pending = [i for i in range(len(items)) if i not in resolved]
    t0 = time.time()
    last_why = {}                           # rejection reasons of the previous attempt, fed back into the retry prompt
    for attempt in range(a.retries):
        if not pending: break
        kof = lambda i: key(model, items[i][0]["task_id"], items[i][1], items[i][2], attempt)
        need = [i for i in pending if kof(i) not in cache]
        if need and a.backend == "cacheonly":
            st["missing_cache"] += len(need)
        elif need:
            groups = {}
            for i in need:
                t, op, lv = items[i]
                groups.setdefault((tok_limit(lv, t["domain"]), round(0.3 + 0.3 * attempt, 2)), []).append(i)
            done = 0
            with open(CACHE, "a", encoding="utf-8") as cf:
                for (tl, temp), idx in groups.items():
                    prompts = {i: build_user_prompt(items[i][0], items[i][1], items[i][2], S.get(items[i][0]["task_id"]), feedback_text(last_why.get(i, []))) for i in idx}
                    idx.sort(key=lambda i: len(prompts[i]))                 # similar lengths together = less padding
                    for b in range(0, len(idx), a.batch_size):
                        chunk = idx[b:b + a.batch_size]
                        outs = backend.generate_batch(SYSTEM, [prompts[i] for i in chunk], a.seed + attempt, tl, temp)
                        for i, o in zip(chunk, outs):
                            cache[kof(i)] = o
                            cf.write(json.dumps({"key": kof(i), "text": o}, ensure_ascii=False) + "\n")
                        cf.flush(); done += len(chunk)
                        print(f"attempt {attempt}: generated {done}/{len(need)}  elapsed {time.time()-t0:.0f}s", flush=True)
        still = []
        for i in pending:
            t, op, lv = items[i]; sol = S.get(t["task_id"])
            cand = clean(cache.get(kof(i), ""))
            if attempt == 0: raw_first[i] = cand
            vc = validate(t, op, lv, cand, sol)
            ok = bool(cand) and not vc["leakage_flag"] and vc["compliance_flag"]
            if attempt == 0 and not ok:
                st["first_pass_violations"] += 1; st["violations_by_level"][lv] += 1
            if ok:
                resolved[i] = (cand, "qwen", vc)
                if attempt > 0: st["regenerated_ok"] += 1
            else:
                last_why[i] = vc["reasons"] + ["LEAK:" + w for w in vc["leak_why"]]
                still.append(i)
        pending = still
    for i in pending:                       # all attempts rejected -> safe template fallback
        t, op, lv = items[i]; sol = S.get(t["task_id"])
        text = fallback_text(t, op, lv, sol)
        resolved[i] = (text, "fallback", validate(t, op, lv, text, sol)); st["fallback"] += 1
    if a.shard:                                   # shard run: only the cache matters, write nothing else
        print(json.dumps(st, indent=1))
        print(f"shard {a.shard} done: {len(items)} items cached")
        return
    rows = []
    for i, (t, op, lv) in enumerate(items):
        text, src, v = resolved[i]
        rows.append({"task_id": t["task_id"], "op": op, "level": lv, "text": text,
                     "leakage_flag": v["leakage_flag"], "compliance_flag": v["compliance_flag"], "source": src})
    with open(ROOT / "data" / "hints.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(CACHE.parent / "raw_first_pass.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(raw_first):
            t, op, lv = items[i]
            f.write(json.dumps({"task_id": t["task_id"], "op": op, "level": lv, "text": raw_first[i]}, ensure_ascii=False) + "\n")
    (CACHE.parent / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))

if __name__ == "__main__":
    main()
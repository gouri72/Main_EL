"""Rule-based validator for hints. Two checks per (task, op, level, text):
  leakage_flag    True = the hint leaks solution/gold label at a level where that is NOT allowed (levels 1-3).
                  Level 4 is full assistance, so leakage_flag is always False there.
  compliance_flag True = the hint satisfies the level rules (length, question form at L1, no code at L1-L2,
                  and at level 4 it really contains the answer).
Solution n-grams that already appear in the task prompt (function name, test asserts) are ignored,
because the learner can already see them."""
import ast, json, re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES = yaml.safe_load(open(ROOT / "assist" / "rules.yaml", encoding="utf-8"))

def toks(s):
    return re.findall(r"[a-z_]\w*|\d+|[^\s\w]", s.lower())

def ngrams(seq, n):
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}

def load_tasks(path=None):
    path = path or ROOT / "data" / "tasks.jsonl"
    return {t["task_id"]: t for t in map(json.loads, open(path, encoding="utf-8")) if t}

def load_solutions(path=None):
    path = path or ROOT / "data" / "solutions.jsonl"
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); out[d["task_id"]] = d["solution"]
    return out

SUP = r"(support(s|ed|ing)?|confirm(s|ed)?|consistent with|is true|are true|correct)"
CON = r"(contradict(s|ed|ing|ion)?|refute(s|d)?|disprove(s|d)?|inconsistent|is false|are false|incorrect)"
HEDGE = r"(\bwhether\b|\bif\b|\bor\b|\?)"
# explicit pointers to a sentence in the abstract ("sentence 3", "3rd sentence", "[2]", "S4") give away the evidence location
SENT_REF = r"(\b(sentence|sentences|line|lines|statement|statements)\s*#?\s*\d+|\b\d+(st|nd|rd|th)\s+(sentence|line|statement)|\[\d+\]|\bS\d+\b)"

def label_leak(text, label):
    """True if some non-hedged sentence asserts the gold polarity."""
    fam = SUP if label == "SUPPORT" else CON
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if re.search(fam, sent, re.I) and not re.search(HEDGE, sent, re.I):
            return True
    return False

ECHO = r"(write the (hint|answer) now|^\s*(hint|answer)\s*:|^\s*task\s*:|\blevel\s*[1-4]\s*\()"

def validate(task, op, level, text, solution=None, rules=RULES):
    r = rules["levels"][level]
    reasons = []
    words = len(text.split())
    if words == 0: reasons.append("empty")
    if words < r.get("min_words", 0): reasons.append("too_short")
    if words > r["max_words"]: reasons.append("too_long")
    if re.search(ECHO, text, re.I | re.M): reasons.append("prompt_echo")
    if r.get("require_answer"):   # level 4 must really contain the answer
        if task["domain"] == "coding":
            if not ("```" in text or re.search(r"\bdef \w+\(", text)): reasons.append("no_answer")
        elif not label_leak(text, task["gold"]["label"]):
            reasons.append("no_answer")
    if r["require_question"]:   # level 1 = one or two questions, nothing else
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sents or not all(s.endswith("?") for s in sents) or len(sents) > 2:
            reasons.append("not_a_question")
    if r["no_code"] and ("```" in text or re.search(r"\bdef \w+\(|\breturn\b.*[=\[(]", text)):
        reasons.append("code_present")
    if "<think>" in text.lower(): reasons.append("thinking_trace")
    if text.count("```") % 2: reasons.append("unclosed_code_block")      # e.g. generation cut off mid-answer
    elif task["domain"] == "coding":
        for blk in re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S):
            try: ast.parse(blk)                                           # syntax only, nothing is executed
            except SyntaxError: reasons.append("code_syntax_error"); break

    leak = False
    if r["enforce_leak"]:
        prompt_toks = toks(task["prompt"])
        if task["domain"] == "coding" and solution:
            n = rules["leak_ngram"]
            novel = ngrams(toks(solution), n) - ngrams(prompt_toks, n)
            leak = len(ngrams(toks(text), n) & novel) > r["max_novel_ngrams"]
        elif task["domain"] == "science":
            leak = label_leak(text, task["gold"]["label"]) or bool(re.search(SENT_REF, text, re.I))
            n = rules["evidence_ngram"]
            ev = set()
            for e in task["gold"]["evidence"]:
                ev |= ngrams(toks(e), n)
            # evidence sentences are in the abstract shown to the learner; copying them is only a leak at L1-L2
            if len(ngrams(toks(text), n) & ev) > rules["evidence_max_novel"][level]:
                leak = True
    return {"leakage_flag": bool(leak), "compliance_flag": not reasons, "reasons": reasons}
#!/usr/bin/env python3
"""Build coding tasks + transfer variants + reference solutions from MBPP.

  python data/scripts/build_coding.py --n 5   --seed 42     # Day-1 sample
  python data/scripts/build_coding.py --n 150 --seed 42     # full set

Writes: data/tasks_coding.jsonl, data/solutions.jsonl  (then run merge_tasks.py)

Transfer variant (near transfer, AUTOMATIC): function renamed (<name>_2) and the integer literals in the
test inputs are shifted by small amounts; expected outputs are recomputed by running the reference
solution. Same logic, new surface. State this in the paper (Data limitations).
Operation labels, difficulty bins and error types are HEURISTIC (label_source = "heuristic").
"""
import argparse, ast, copy, json, os, random, re, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mbpp.jsonl"
OUT_TASKS = ROOT / "data" / "tasks_coding.jsonl"
OUT_SOLS = ROOT / "data" / "solutions.jsonl"
BANNED = re.compile(r"\b(input|open|exec|eval|__import__|compile)\s*\(|import\s+(os|sys|subprocess|socket|shutil|requests)")


def run_py(code, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    finally:
        os.unlink(path)


def parse_asserts(tests):
    """Return (list of Call nodes, function name) if every test is `assert f(...) == expected`."""
    calls, expected = [], []
    for s in tests:
        try:
            tree = ast.parse(s.strip())
        except SyntaxError:
            return None
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            return None
        t = tree.body[0].test
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
                and isinstance(t.left, ast.Call) and isinstance(t.left.func, ast.Name)):
            return None
        calls.append(t.left)
        expected.append(ast.unparse(t.comparators[0]))
    names = {c.func.id for c in calls}
    return (calls, names.pop(), expected) if len(names) == 1 else None


class IntBump(ast.NodeTransformer):
    def __init__(self, rng):
        self.rng, self.changed = rng, 0

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, int) and not isinstance(v, bool):
            d = self.rng.choice([-2, -1, 1, 2, 3])
            nv = v + d
            if v >= 0 and nv < 0:
                nv = v + abs(d)
            if nv != v:
                self.changed += 1
                return ast.copy_location(ast.Constant(nv), node)
        return node


def make_variant(solution, name, calls, rng, orig_distinct):
    newname = name + "_2"
    mutated = []
    for call in calls:
        c = copy.deepcopy(call)
        bump = IntBump(rng)
        c.args = [bump.visit(a) for a in c.args]
        c.keywords = [bump.visit(k) for k in c.keywords]
        if bump.changed:
            mutated.append(c)
    if len(mutated) < 2:
        return None
    exprs = [ast.unparse(c) for c in mutated]
    harness = (solution + "\n\nimport json as _j\n_exprs = " + repr(exprs) +
               "\n_out = []\nfor _e in _exprs:\n    try:\n        _out.append(repr(eval(_e)))\n"
               "    except Exception:\n        _out.append(None)\nprint('@@RES@@' + _j.dumps(_out))\n")
    ok, out, _ = run_py(harness)
    if not ok:
        return None
    line = [l for l in out.splitlines() if l.startswith("@@RES@@")]
    if not line:
        return None
    results = json.loads(line[0][7:])
    tests = []
    for c, res in zip(mutated, results):
        if res is None or len(res) > 150 or any(b in res for b in ("nan", "inf", "<")):
            continue
        try:
            ast.literal_eval(res)
        except Exception:
            continue
        c.func = ast.Name(newname, ast.Load())
        tests.append(f"assert {ast.unparse(c)} == {res}")
    if len(tests) < 2:
        return None
    # avoid degenerate variants (e.g. every expected value identical) when the original has variety
    if len({t.split(" == ", 1)[1] for t in tests}) < min(orig_distinct, 2):
        return None
    var_sol = re.sub(rf"\b{re.escape(name)}\b", newname, solution)
    ok, _, _ = run_py(var_sol + "\n\n" + "\n".join(tests) + "\n")
    if not ok:
        return None
    return {"solution": var_sol, "tests": tests, "name": newname}


def error_types_for(solution):
    types = set()
    for n in ast.walk(ast.parse(solution)):
        if isinstance(n, (ast.For, ast.While)):
            types.add("loop_logic")
        if isinstance(n, ast.Subscript) or (isinstance(n, ast.Call) and getattr(n.func, "id", "") == "range"):
            types.add("off_by_one")
        if isinstance(n, ast.If):
            types.add("wrong_condition")
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            types.add("type_handling")
        if isinstance(n, ast.BinOp):
            types.add("wrong_formula")
    return sorted(types) or ["wrong_formula"]


def operations_for(difficulty):
    ops = ["understand_plan", "execute"]
    if difficulty >= 2:
        ops.append("debug_verify")
    if difficulty >= 3:
        ops.append("explain")
    return [{"op": o, "label_source": "heuristic"} for o in ops]


def prompt_for(text, first_test):
    return f"{text.strip()}\nYour code should pass this test:\n{first_test.strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if not RAW.exists():
        sys.exit(f"{RAW} not found. Run: python data/scripts/download_data.py")

    with open(RAW, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    random.Random(a.seed).shuffle(rows)

    accepted, rej = [], Counter()
    for r in rows:
        if len(accepted) >= a.n:
            break
        code = r["code"].replace("\r\n", "\n").replace("\r", "\n").rstrip()
        tests = [t.strip() for t in r["test_list"]]
        if r.get("test_setup_code", "").strip():
            rej["setup_code"] += 1; continue
        if re.search(r"\d", r["text"]):
            rej["digits_in_text"] += 1; continue
        if BANNED.search(code) or len(code.splitlines()) > 40:
            rej["unsafe_or_long"] += 1; continue
        parsed = parse_asserts(tests)
        if not parsed:
            rej["test_format"] += 1; continue
        calls, name, expected = parsed
        if not re.search(rf"def\s+{re.escape(name)}\s*\(", code):
            rej["no_def"] += 1; continue
        ok, _, _ = run_py(code + "\n\n" + "\n".join(tests) + "\n")
        if not ok:
            rej["solution_fails"] += 1; continue
        rng = random.Random(f"{a.seed}-{r['task_id']}")
        var = make_variant(code, name, calls, rng, len(set(expected)))
        if not var:
            rej["no_variant"] += 1; continue
        accepted.append({"row": r, "code": code, "tests": tests, "var": var,
                         "nodes": sum(1 for _ in ast.walk(ast.parse(code)))})

    # difficulty = tertile of solution AST size
    order = sorted(range(len(accepted)), key=lambda i: accepted[i]["nodes"])
    for rank, i in enumerate(order):
        accepted[i]["difficulty"] = 1 + min(2, rank * 3 // max(1, len(accepted)))

    tasks, sols = [], []
    for k, x in enumerate(accepted, 1):
        tid = f"code_{k:03d}"
        d = x["difficulty"]
        ops = operations_for(d)
        et = error_types_for(x["code"])
        t_tests = "\n".join(x["tests"])
        v_tests = "\n".join(x["var"]["tests"])
        tasks.append({"task_id": tid, "domain": "coding",
                      "prompt": prompt_for(x["row"]["text"], x["tests"][0]),
                      "difficulty": d, "operations": ops, "gold": {"tests": t_tests},
                      "transfer_task_id": tid + "_T", "error_types": et})
        tasks.append({"task_id": tid + "_T", "domain": "coding",
                      "prompt": prompt_for(x["row"]["text"], x["var"]["tests"][0]),
                      "difficulty": d, "operations": ops, "gold": {"tests": v_tests},
                      "transfer_task_id": tid + "_T", "error_types": et})
        sols.append({"task_id": tid, "solution": x["code"]})
        sols.append({"task_id": tid + "_T", "solution": x["var"]["solution"]})

    with open(OUT_TASKS, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(OUT_SOLS, "w", encoding="utf-8", newline="\n") as f:
        for s in sols:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(accepted)} tasks (+{len(accepted)} variants) -> {OUT_TASKS.name}, {OUT_SOLS.name}")
    print("difficulty:", dict(sorted(Counter(x['difficulty'] for x in accepted).items())))
    print("rejected while scanning:", dict(rej))
    if len(accepted) < a.n:
        print(f"WARNING: only {len(accepted)} of {a.n} requested passed the filters")


if __name__ == "__main__":
    main()

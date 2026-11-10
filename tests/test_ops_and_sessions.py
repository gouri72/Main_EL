"""Tests for operation labels and the session sampler. Works with pytest or as a script:
  python tests\test_ops_and_sessions.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.loader import load_tasks, session_pool, draw_session, abstract_key

ALL = load_tasks()
POOL = session_pool(ALL)


def test_coding_ops_not_determined_by_difficulty():
    by = defaultdict(set)
    for t in POOL:
        if t["domain"] == "coding":
            by[t["difficulty"]].add(tuple(o["op"] for o in t["operations"]))
    for d in (1, 2, 3):
        assert len(by[d]) >= 3, "difficulty %d has only %d distinct op lists" % (d, len(by[d]))


def test_every_task_has_plan_and_execute():
    for t in ALL.values():
        ops = [o["op"] for o in t["operations"]]
        assert ops[:2] == ["understand_plan", "execute"], t["task_id"]


def test_variant_ops_match_base():
    for t in POOL:
        v = ALL[t["transfer_task_id"]]
        assert [o["op"] for o in v["operations"]] == [o["op"] for o in t["operations"]], t["task_id"]


def test_science_variant_uses_different_abstract():
    for t in POOL:
        if t["domain"] == "science":
            assert abstract_key(t) != abstract_key(ALL[t["transfer_task_id"]]), t["task_id"]


def test_draw_session_properties():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        s = draw_session(POOL, rng, n=8)
        assert len(s) == 8 and len({t["task_id"] for t in s}) == 8
        assert len({abstract_key(t) for t in s}) == 8, "shared abstract in session"
        assert all(not t["task_id"].endswith("_T") for t in s)
        counts = [sum(t["difficulty"] == d for t in s) for d in (1, 2, 3)]
        assert max(counts) - min(counts) <= 1, counts


def test_draw_session_deterministic_and_domain_filter():
    a = [t["task_id"] for t in draw_session(POOL, np.random.default_rng(3), 8)]
    b = [t["task_id"] for t in draw_session(POOL, np.random.default_rng(3), 8)]
    assert a == b
    s = draw_session(POOL, np.random.default_rng(1), 8, domain="science")
    assert all(t["domain"] == "science" for t in s)


if __name__ == "__main__":
    ok = True
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok", name)
            except AssertionError as e:
                ok = False; print("FAIL", name, e)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
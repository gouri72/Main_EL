"""Sanity tests for the P3 data layer. From the repo root:  python -m pytest tests/test_data.py   or   python tests/test_data.py"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.loader import load_tasks, session_pool, transfer_of  # noqa: E402


def test_counts_and_pairs():
    tasks = load_tasks()
    pool = session_pool(tasks)
    assert len(tasks) == 600 and len(pool) == 300
    for dom in ("coding", "science"):
        assert sum(t["domain"] == dom for t in pool) == 150
    for t in pool:
        v = transfer_of(tasks, t)
        assert v["task_id"] == t["task_id"] + "_T"
        assert v["domain"] == t["domain"] and v["difficulty"] == t["difficulty"]
        assert v["prompt"] != t["prompt"]


def test_session_pool_excludes_variants():
    assert all(not t["task_id"].endswith("_T") for t in session_pool(load_tasks()))


def test_all_four_operations_present():
    ops = {o["op"] for t in session_pool(load_tasks()) for o in t["operations"]}
    assert ops == {"understand_plan", "execute", "debug_verify", "explain"}


def test_validator_passes_and_solutions_run():
    r = subprocess.run([sys.executable, str(ROOT / "data" / "validate_tasks.py"), "--run-tests"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:]


def test_data_checks_pass():
    r = subprocess.run([sys.executable, str(ROOT / "data" / "scripts" / "data_checks.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
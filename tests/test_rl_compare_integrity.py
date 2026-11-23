"""Safety checks for reproducible frozen-policy experiment comparisons."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "analysis" / "compare.py"
spec = importlib.util.spec_from_file_location("compare_guarded", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write_result(path, seeds=(0, 1), n=3, duplicate=False):
    path.mkdir()
    episodes = []
    steps = []
    for seed in seeds:
        for ep in range(n):
            episodes.append(dict(
                seed=seed, phase="eval", episode=ep,
                post_transfer=0.3, transfer_gain=0.02,
                assist_spent=2, full_assist_rate=0,
                first_attempt_rate=1, takeover_rate=0
            ))
            steps.append(dict(
                seed=seed, phase="eval", episode=ep,
                task_idx=0, success=True
            ))
    if duplicate:
        episodes.append(dict(episodes[0]))
    pd.DataFrame(episodes).to_csv(path / "episodes.csv", index=False)
    pd.DataFrame(steps).to_csv(path / "steps.csv", index=False)
    return path


def test_smoke_run_cannot_pass_large_k(tmp_path):
    d = write_result(tmp_path / "smoke", seeds=(0,), n=2)
    with pytest.raises(ValueError, match="fewer than 100 evaluation episodes"):
        mod.per_seed(d, 100)


def test_exact_k_accepted(tmp_path):
    d = write_result(tmp_path / "full", n=3)
    summary = mod.per_seed(d, 3)
    assert len(summary) == 2
    assert all(summary.immediate_success == 1)


def test_duplicate_eval_episode_rejected(tmp_path):
    d = write_result(tmp_path / "duplicate", duplicate=True)
    with pytest.raises(ValueError, match="duplicate frozen evaluation"):
        mod.per_seed(d, 2)


def test_duplicate_eval_step_rejected(tmp_path):
    d = write_result(tmp_path / "steps")
    s = pd.read_csv(d / "steps.csv")
    pd.concat([s, s.iloc[[0]]], ignore_index=True).to_csv(
        d / "steps.csv", index=False
    )
    with pytest.raises(ValueError, match="duplicate evaluation step"):
        mod.per_seed(d, 2)

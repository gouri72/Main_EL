"""Regression: RL comparison must use held-out evaluation, not training episodes."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[1] / "analysis" / "compare.py"
spec = importlib.util.spec_from_file_location("rl_compare", MODULE)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def test_frozen_eval_only(tmp_path):
    d = tmp_path / "experiment"
    d.mkdir()
    episode_rows = []
    step_rows = []
    for seed in (0, 1):
        for phase in ("train", "eval"):
            for ep in (0, 1, 2):
                success = 1 if phase == "train" else 0
                episode_rows.append(dict(seed=seed, phase=phase, episode=ep,
                    post_transfer=1.0 if phase == "train" else 0.2 + ep * 0.1,
                    transfer_gain=1.0 if phase == "train" else 0.1,
                    assist_spent=4, full_assist_rate=0.5, first_attempt_rate=0.5,
                    takeover_rate=0.2))
                step_rows.append(dict(seed=seed, phase=phase, episode=ep,
                                      success=success))
    pd.DataFrame(episode_rows).to_csv(d / "episodes.csv", index=False)
    pd.DataFrame(step_rows).to_csv(d / "steps.csv", index=False)
    actual = compare.per_seed(d, 2)
    assert list(actual.index) == [0, 1]
    assert all(actual["immediate_success"] == 0)
    assert all(actual["post_transfer"].round(4) == 0.35)
    assert all(actual["transfer_gain"] == 0.1)


def test_rejects_no_frozen_evaluation(tmp_path):
    d = tmp_path / "train_only"
    d.mkdir()
    pd.DataFrame([dict(seed=0, phase="train", episode=0, post_transfer=0.1,
                       transfer_gain=0.1, assist_spent=0, full_assist_rate=0,
                       first_attempt_rate=0, takeover_rate=0)]).to_csv(d / "episodes.csv", index=False)
    pd.DataFrame([dict(seed=0, phase="train", episode=0, success=1)]).to_csv(d / "steps.csv", index=False)
    with pytest.raises(ValueError, match="no evaluation"):
        compare.per_seed(d, 2)

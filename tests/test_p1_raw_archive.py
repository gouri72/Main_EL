"""Regression tests for lossless, fail-closed P1 original-result archiving."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml
from tools.archive_p1_results import archive, validate_run


def fixture_run(root, name="tabular_q", seeds=(0, 1), n_train=2, n_eval=3, n_tasks=2):
    exp_id = f"A_novice_v2__{name}"
    config = {
        "exp_id": exp_id, "seeds": list(seeds), "n_episodes": n_train,
        "n_eval_episodes": n_eval, "data": {"session_tasks": n_tasks}
    }
    folder = root / "results" / exp_id
    folder.mkdir(parents=True)
    config_path = root / "configs" / "grid" / f"{exp_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(config, sort_keys=False)
    config_path.write_text(content, encoding="utf-8")
    (folder / "config.yaml").write_text(content, encoding="utf-8")
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    episodes = []
    steps = []
    for seed in seeds:
        for phase, n in (("train", n_train), ("eval", n_eval)):
            for ep in range(n):
                episodes.append(dict(seed=seed, phase=phase, episode=ep))
                for task_idx in range(n_tasks):
                    steps.append(dict(seed=seed, phase=phase, episode=ep, task_idx=task_idx))
    for filename, rows in (("episodes.csv", episodes), ("steps.csv", steps)):
        with (folder / filename).open("w", newline="", encoding="utf-8") as f:
            f.write(f"# config_hash={h} exp_id={exp_id}\n")
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
    return folder


def test_validate_and_archive_original_rows(tmp_path):
    fixture_run(tmp_path)
    report = validate_run(tmp_path, "tabular_q")
    assert report["episode_rows"] == 10
    assert report["step_rows"] == 20
    out = tmp_path / "artifacts" / "evidence.zip"
    archive(tmp_path, out, run_names=("tabular_q",))
    with ZipFile(out) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read("P1_EVIDENCE_MANIFEST.json"))
        assert manifest["runs"][0]["step_rows"] == 20
        for path, meta in manifest["runs"][0]["files"].items():
            assert hashlib.sha256(z.read(path)).hexdigest() == meta["sha256"]
    with pytest.raises(FileExistsError):
        archive(tmp_path, out, run_names=("tabular_q",))


def test_missing_episode_fails_without_archive(tmp_path):
    folder = fixture_run(tmp_path)
    file = folder / "episodes.csv"
    lines = file.read_text().splitlines(keepends=True)
    file.write_text("".join(lines[:-1]))
    out = tmp_path / "absent.zip"
    with pytest.raises(ValueError, match="incomplete or excess"):
        archive(tmp_path, out, run_names=("tabular_q",))
    assert not out.exists()


def test_duplicate_step_fails(tmp_path):
    folder = fixture_run(tmp_path)
    path = folder / "steps.csv"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines) + lines[-1])
    with pytest.raises(ValueError, match="duplicate row"):
        validate_run(tmp_path, "tabular_q")


def test_modified_config_rejected(tmp_path):
    folder = fixture_run(tmp_path)
    (folder / "config.yaml").write_text("exp_id: modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config differs"):
        validate_run(tmp_path, "tabular_q")


def test_missing_required_run_fails_closed(tmp_path):
    fixture_run(tmp_path)
    with pytest.raises(FileNotFoundError):
        archive(tmp_path, tmp_path / "out.zip", run_names=("tabular_q", "linear_q"))
    assert not (tmp_path / "out.zip").exists()

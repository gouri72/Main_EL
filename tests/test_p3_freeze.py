"""Regression tests for P3 dataset freeze manifest.

Run: python -m pytest -q tests/test_p3_freeze.py
"""
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "data" / "scripts" / "freeze_manifest.py"
spec = importlib.util.spec_from_file_location("p3_freeze_manifest", SRC)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


def test_snapshot_deterministic(tmp_path):
    for rel in freeze.FILES:
        (tmp_path / rel).write_text('{"k":1}\n' if rel.endswith(".jsonl") else "seed: 42\n", encoding="utf-8")
    first = freeze.snapshot(tmp_path)
    second = freeze.snapshot(tmp_path)
    assert first == second
    assert first["files"]["tasks.jsonl"]["records"] == 1
    assert not freeze.verify(first, second)


def test_detects_changed_file(tmp_path):
    for rel in freeze.FILES:
        (tmp_path / rel).write_text("original\n", encoding="utf-8")
    initial = freeze.snapshot(tmp_path)
    (tmp_path / "tasks.jsonl").write_text("modified\n", encoding="utf-8")
    current = freeze.snapshot(tmp_path)
    issues = freeze.verify(initial, current)
    assert len(issues) == 1
    assert "tasks.jsonl" in issues[0]


def test_rejects_incomplete_manifest(tmp_path):
    for rel in freeze.FILES:
        (tmp_path / rel).write_text("original\n", encoding="utf-8")
    initial = freeze.snapshot(tmp_path)
    initial["files"].pop("hints.jsonl")
    issues = freeze.verify(initial, freeze.snapshot(tmp_path))
    assert any("hints.jsonl" in x for x in issues)


def test_missing_artifact_fails(tmp_path):
    try:
        freeze.snapshot(tmp_path)
        assert False, "must reject missing dataset files"
    except FileNotFoundError:
        pass

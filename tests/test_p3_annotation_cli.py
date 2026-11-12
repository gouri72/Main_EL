"""Tests for independent, resumable P3 human annotation CLI."""
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "scripts" / "annotate_tasks.py"
spec = importlib.util.spec_from_file_location("p3_human_annotation", SRC)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_fixed_sample_question_parser():
    blocks = tool.question_blocks()
    ids = tool.sample()
    assert len(ids) == len(blocks) == 40
    assert {v[0] for v in blocks.values()} == {"coding", "science"}
    assert "parabola" in blocks["code_021"][1].lower()


def test_init_has_no_filled_in_ratings(tmp_path):
    ids = tool.sample()
    blocks = tool.question_blocks()
    rows = tool.new_rows(ids, blocks)
    sheet = tmp_path / "human.csv"
    tool.save_rows(sheet, rows, ids)
    reloaded = tool.read_rows(sheet)
    tool.validate_rows(reloaded, blocks, ids)
    assert tool.status(reloaded, ids)[:3] == (0, 0, 0)
    assert all(not tool.completed(r) for r in reloaded.values())


def test_completed_rows_are_preserved(tmp_path):
    ids, blocks = tool.sample(), tool.question_blocks()
    rows = tool.new_rows(ids, blocks)
    first = ids[0]
    rows[first].update(dict(understand_plan="1", execute="1",
                            debug_verify="0", explain="0", difficulty="2"))
    sheet = tmp_path / "progress.csv"
    tool.save_rows(sheet, rows, ids)
    reread = tool.read_rows(sheet)
    assert tool.completed(reread[first])
    assert tool.status(reread, ids)[0] == 1
    assert reread[first]["difficulty"] == "2"


def test_invalid_rating_rejected(tmp_path):
    ids, blocks = tool.sample(), tool.question_blocks()
    rows = tool.new_rows(ids, blocks)
    rows[ids[0]]["difficulty"] = "4"
    sheet = tmp_path / "invalid.csv"
    tool.save_rows(sheet, rows, ids)
    with pytest.raises(ValueError, match="difficulty"):
        tool.read_rows(sheet)


def test_missing_and_duplicate_rows_rejected(tmp_path):
    ids, blocks = tool.sample(), tool.question_blocks()
    rows = tool.new_rows(ids, blocks)
    rows.pop(ids[-1])
    with pytest.raises(ValueError, match="40 fixed IDs"):
        tool.validate_rows(rows, blocks, ids)
    sheet = tmp_path / "dupes.csv"
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tool.FIELDS)
        writer.writeheader()
        writer.writerow(tool.new_rows(ids, blocks)[ids[0]])
        writer.writerow(tool.new_rows(ids, blocks)[ids[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        tool.read_rows(sheet)


def test_malformed_domain_rejected():
    ids, blocks = tool.sample(), tool.question_blocks()
    rows = tool.new_rows(ids, blocks)
    rows[ids[0]]["domain"] = "incorrect"
    with pytest.raises(ValueError, match="wrong domain"):
        tool.validate_rows(rows, blocks, ids)

import csv
import sys

import pytest

from assist.ft.prepare_two_rater_sheets import main as prepare_sheets
from assist.ft.prepare_error_review_sheets import main as prepare_error_sheets
from assist.ft.summarize_human_ratings import kappa


def test_prepare_two_blind_sheets_keeps_pairs_and_adds_usefulness(tmp_path, monkeypatch):
    source = tmp_path / "blind_annotation_sheet.csv"
    fields = ["rating_id", "domain", "operation", "level", "task_input", "output_A", "output_B",
              "A_relevance_1to5", "B_relevance_1to5", "A_correctness_1to5", "B_correctness_1to5",
              "A_level_fit_1to5", "B_level_fit_1to5", "preference_A_B_tie", "notes"]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"rating_id": "r1", "domain": "coding", "operation": "execute", "level": 2,
                         "task_input": "task", "output_A": "hint A", "output_B": "hint B"})
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_two_rater_sheets", "--sheet", str(source), "--output-dir", str(out)])
    prepare_sheets()
    first, second = [list(csv.DictReader(path.open(encoding="utf-8", newline="")))
                     for path in (out / "blind_rater_1.csv", out / "blind_rater_2.csv")]
    assert first[0]["rating_id"] == second[0]["rating_id"] == "r1"
    assert first[0]["output_A"] == second[0]["output_A"] == "hint A"
    assert "A_usefulness_1to5" in first[0]
    assert first[0]["A_usefulness_1to5"] == ""
    assert first[0]["preference_A_B_tie"] == ""


def test_weighted_kappa_reports_perfect_agreement():
    result = kappa([1, 2, 4, 5], [1, 2, 4, 5], quadratic=True)
    assert result["agreement"] == 1
    assert result["kappa"] == 1


def test_weighted_kappa_reports_disagreement():
    result = kappa([1, 1, 5, 5], [5, 5, 1, 1], quadratic=True)
    assert result["agreement"] == 0
    assert result["kappa"] == -1


def test_error_review_forms_hide_model_and_automatic_case_category(tmp_path, monkeypatch):
    cases = tmp_path / "paired_error_cases.csv"
    fields = ["example_id", "task_id", "group_id", "domain", "op", "level", "paired_case",
              "base_failure_categories", "lora_failure_categories", "base_output", "lora_output"]
    with cases.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"example_id": "e1", "task_id": "t1", "group_id": "g1", "domain": "science",
                         "op": "explain", "level": 2, "paired_case": "both_pass",
                         "base_output": "base text", "lora_output": "lora text"})
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"example_id":"e1","user":"claim and evidence context"}\n', encoding="utf-8")
    out = tmp_path / "review"
    monkeypatch.setattr(sys, "argv", ["prepare_error_review_sheets", "--cases", str(cases),
                                       "--test-file", str(test_file), "--output-dir", str(out)])
    prepare_error_sheets()
    rater1 = list(csv.DictReader((out / "error_review_rater_1.csv").open(encoding="utf-8", newline="")))
    key = list(csv.DictReader((out / "error_review_key.csv").open(encoding="utf-8", newline="")))
    assert len(rater1) == 1
    assert rater1[0]["task_context"] == "claim and evidence context"
    assert "automatic_category" not in rater1[0]
    assert "_model" not in " ".join(rater1[0])
    assert key[0]["automatic_category"] == "both_pass"


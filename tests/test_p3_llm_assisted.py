"""Regression tests for the exploratory LLM-assisted audit."""
import importlib.util
import csv
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "data" / "scripts" / "audit_llm_assisted.py"
spec = importlib.util.spec_from_file_location("p3_llm_assisted", SRC)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def complete_rows(model="Qwen3-0.6B"):
    ids, domains = audit.ids_and_domains()
    return [{**{"task_id": tid, "domain": domains[tid]},
             **{op: "1" for op in audit.OPS},
             "difficulty": "2", "model_id": model,
             "review_status": "model_only", "notes": ""}
            for tid in ids]


def test_blank_template_has_no_model_annotations():
    template = Path(__file__).resolve().parents[1] / "data" / "audit" / "annotations_template.csv"
    with template.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40
    assert all(not r[c] for r in rows for c in (*audit.OPS, "difficulty"))


def test_complete_llm_labels_validate():
    ids, domains = audit.ids_and_domains()
    audit.validate_labels(complete_rows(), ids, domains, allow_incomplete=False)


def test_duplicate_task_ids_rejected():
    ids, domains = audit.ids_and_domains()
    rows = complete_rows()
    rows[1]["task_id"] = rows[0]["task_id"]
    with pytest.raises(ValueError):
        audit.validate_labels(rows, ids, domains, allow_incomplete=False)


def test_model_provenance_required():
    ids, domains = audit.ids_and_domains()
    rows = complete_rows()
    rows[0]["model_id"] = ""
    with pytest.raises(ValueError, match="model_id"):
        audit.validate_labels(rows, ids, domains, allow_incomplete=False)


def test_invalid_labels_and_review_status_rejected():
    ids, domains = audit.ids_and_domains()
    rows = complete_rows()
    rows[0]["debug_verify"] = "2"
    with pytest.raises(ValueError, match="labels"):
        audit.validate_labels(rows, ids, domains, allow_incomplete=False)
    rows[0]["debug_verify"] = "0"
    rows[0]["review_status"] = "independent_human"
    with pytest.raises(ValueError, match="review_status"):
        audit.validate_labels(rows, ids, domains, allow_incomplete=False)


def test_report_is_explicitly_not_human_reliability():
    rendered = audit.report(complete_rows())
    assert "NOT an independent human annotation study" in rendered
    assert "| coding | 20 |" in rendered
    assert "| science | 20 |" in rendered

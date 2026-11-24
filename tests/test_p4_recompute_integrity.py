"""CPU-only validation guards for raw P4 output recomputation."""
import pytest
from assist.ft.recompute_lora_summary import validate_paired_rows
from assist.ft.evaluate_lora import digest


def example():
    test = {"example_id": "x", "task_id": "t", "group_id": "g", "domain": "coding",
            "op": "execute", "level": 2, "system": "rules", "user": "problem"}
    row = {k: test[k] for k in ("example_id", "task_id", "group_id", "domain", "op", "level")}
    row.update(task_input=test["user"], prompt_hash=digest(test["system"] + test["user"]),
               base={"cleaned_text": "a", "compliance_flag": True, "leakage_flag": False,
                     "latency_seconds": 1.0},
               lora={"cleaned_text": "b", "compliance_flag": True, "leakage_flag": False,
                     "latency_seconds": 1.0})
    return test, row


def test_matches_exact_test_prompt_and_metadata():
    test, row = example()
    assert validate_paired_rows([row], [test])


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(task_input="other"),
    lambda r: r.update(prompt_hash="bad"),
    lambda r: r.update(group_id="wrong"),
    lambda r: r.update(level=3),
    lambda r: r.pop("lora"),
])
def test_rejects_tampered_paired_output(mutate):
    test, row = example()
    mutate(row)
    with pytest.raises(ValueError):
        validate_paired_rows([row], [test])


def test_rejects_partial_test_set():
    test, row = example()
    with pytest.raises(ValueError, match="incomplete"):
        validate_paired_rows([], [test])

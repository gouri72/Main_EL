"""P3 quality and agreement regression checks. Run: python -m pytest -q tests/test_p3_quality.py"""
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data" / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_audit_is_pure_and_flags_bad_pairs():
    mod = load_script("audit_p3_quality")
    base = {"task_id": "code_001", "domain": "coding", "difficulty": 1,
            "prompt": "implement f", "operations": [{"op": "execute"}],
            "gold": {"tests": "assert f(1)==1"}, "transfer_task_id": "code_001_T"}
    variant = dict(base, task_id="code_001_T", prompt="implement f2",
                   gold={"tests": "assert f2(2)==2"}, transfer_task_id="code_001_T")
    tasks = {"code_001": base, "code_001_T": variant}
    before = json.dumps(tasks, sort_keys=True)
    report = mod.analyze(tasks)
    assert not report["hard"], report["hard"]
    assert json.dumps(tasks, sort_keys=True) == before
    assert "Coding session tasks: 1" in mod.render(report)
    variant["gold"] = {"tests": "assert f(1)==1"}
    assert "identical_test_lines" in mod.analyze(tasks)["hard"]


def test_quality_audit_rejects_duplicate_ids():
    mod = load_script("audit_p3_quality")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "data.jsonl"
        line = json.dumps({"task_id": "t1"})
        p.write_text(line + "\n" + line + "\n", encoding="utf-8")
        try:
            mod.read_jsonl(p)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "duplicate" in str(exc)


def test_agreement_rejects_incomplete_and_duplicates():
    mod = load_script("audit_agreement")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "sheet.csv"
        header = "task_id,domain,understand_plan,execute,debug_verify,explain,difficulty\n"
        p.write_text(header + "t1,coding,,,,,\n", encoding="utf-8")
        try:
            mod.read_sheet(p)
            assert False, "expected incomplete sheet rejection"
        except ValueError:
            pass
        row = "t1,coding,1,1,0,0,1\n"
        p.write_text(header + row + row, encoding="utf-8")
        try:
            mod.read_sheet(p)
            assert False, "expected duplicate rejection"
        except ValueError as exc:
            assert "duplicate" in str(exc)


def test_agreement_rejects_different_sample_sets():
    mod = load_script("audit_agreement")
    row = {"domain": "coding", "diff": 1, "execute": 1,
           "understand_plan": 1, "debug_verify": 0, "explain": 0}
    try:
        mod.validate_sheets({"a": row}, {"b": row}, {"a": row, "b": row}, ["a", "b"])
        assert False, "expected mismatched sample rejection"
    except ValueError as exc:
        assert "exactly" in str(exc)


def test_agreement_kappa_constant_case():
    mod = load_script("audit_agreement")
    assert mod.kappa([1, 1], [1, 1]) is None
    assert mod.kappa([0, 1], [0, 1]) == 1.0

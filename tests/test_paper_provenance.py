"""CI regression for integrated paper numerical provenance."""
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "paper/check_reported_results.py"
spec = importlib.util.spec_from_file_location("reported_results", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_manuscript_numbers_match_committed_sources():
    mod.verify()

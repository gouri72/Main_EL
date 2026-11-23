"""Safety regression tests for the research handoff ZIP."""
import hashlib
import json
from zipfile import ZipFile
import pytest
from tools.build_research_handoff import package

def test_sources_manifest_and_no_overwrite(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "paper.md").write_text("simulation-only draft")
    dest = tmp_path / "release.zip"
    manifest = package(root, dest, files=("paper.md",))
    assert manifest["original_p1_evidence"]["included"] is False
    with ZipFile(dest) as z:
        assert z.testzip() is None
        assert "HANDOFF_README.md" in z.namelist()
        record = json.loads(z.read("RELEASE_MANIFEST.json"))
        assert hashlib.sha256(z.read("paper.md")).hexdigest() == record["source_files"]["paper.md"]["sha256"]
    with pytest.raises(FileExistsError):
        package(root, dest, files=("paper.md",))

def test_missing_input_rejected_before_writing(tmp_path):
    destination = tmp_path / "missing.zip"
    with pytest.raises(FileNotFoundError):
        package(tmp_path, destination, files=("unknown.txt",))
    assert not destination.exists()

@pytest.mark.parametrize("relative", ("../secret", "/abs"))
def test_path_escape_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        package(tmp_path, tmp_path / "out.zip", files=(relative,))

def test_fake_p1_archive_is_rejected(tmp_path):
    (tmp_path / "source.md").write_text("source")
    fake = tmp_path / "fake.zip"
    with ZipFile(fake, "w") as z:
        z.writestr("not_evidence", "fake")
    out = tmp_path / "release.zip"
    with pytest.raises(ValueError, match="lacks original-run"):
        package(tmp_path, out, files=("source.md",), p1_archive=fake)
    assert not out.exists()

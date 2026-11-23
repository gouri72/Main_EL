#!/usr/bin/env python3
"""Bundle an accurately labeled research handoff; never overwrite source files.

The release is for review, not a venue-ready submission. Optional original
P1 evidence is checked against its embedded per-file SHA-256 manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "paper/manuscript.md", "paper/evidence_inventory.md", "paper/datasets.md",
    "paper/p3_results.md", "paper/related_work_p3.md",
    "README_P1.md", "README_P2.md", "README_P3.md", "README_P4_FT.md",
    "results/summary_A_novice_v2.md",
    "results/summary_A_novice_v2_ablations.md",
    "results/person2_cross_family_summary.csv",
    "results/person2_robustness_summary.csv",
    "results/qwen3_06b_lora_v1/RESULTS.md",
    "results/qwen3_06b_lora_v1/evaluation/summary.json",
    "data/audit/llm_assisted_report.md",
    "data/stats/dataset_stats.md",
    "data/stats/p3_freeze_manifest.json",
    "paper/figs/fading.pdf", "paper/figs/pareto.pdf",
    "paper/figs/transfer_bar.pdf",
    "results/person2_figures/cross_family_transfer.png",
    "results/person2_figures/practice_success_by_family.png",
    "results/person2_figures/profile_transfer_comparison.png",
    "results/person2_figures/transfer_by_family.png",
    "results/qwen3_06b_lora_v1/figures/acceptance_by_stratum.svg",
    "results/qwen3_06b_lora_v1/figures/base_vs_lora_overall.svg",
)
NOTICE = """# Research handoff — NOT a published or submission-ready manuscript

This package contains a paper draft, recorded data summaries and figures.
Human P3 and P4 rating studies have not been completed. P2 cross-family
training-seed replication is incomplete; P4 raw outputs/weights are external.
Synthetic learner transfer is not human performance. Authors must approve
claims, authorship, references, figures and venue formatting before submission.
See paper/evidence_inventory.md.
"""

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def package(root, output, files=FILES, p1_archive=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    if len(set(files)) != len(files):
        raise ValueError("Duplicate source paths")
    inputs = []
    for rel in files:
        # Host Path.is_absolute() is insufficient: on Windows '/abs' has
        # no drive and can be reported as non-absolute, while on Linux
        # 'C:\\secret' is just a relative POSIX filename.
        # Archive names must use POSIX-style relative paths on every OS.
        posix = PurePosixPath(rel)
        windows = PureWindowsPath(rel)
        if (not isinstance(rel, str) or not rel or
                posix.is_absolute() or windows.anchor or
                "\\\\" in rel or ".." in posix.parts or
                "." in posix.parts or rel.startswith(("/", "\\\\"))):
            raise ValueError(f"Unsafe relative path: {rel}")
        part = Path(rel)
        source = root / part
        if not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(root):
            raise FileNotFoundError(f"Missing/unsafe release source: {source}")
        inputs.append((str(part).replace("\\", "/"), source))
    p1 = None
    if p1_archive is not None:
        p1 = Path(p1_archive).resolve()
        if not p1.is_file() or p1.is_symlink():
            raise FileNotFoundError(f"Missing original P1 evidence: {p1}")
        with ZipFile(p1) as archive:
            names = set(archive.namelist())
            if "P1_EVIDENCE_MANIFEST.json" not in names:
                raise ValueError("P1 ZIP lacks original-run evidence manifest")
            manifest = json.loads(archive.read("P1_EVIDENCE_MANIFEST.json"))
            if len(manifest.get("runs", [])) != 12 or archive.testzip() is not None:
                raise ValueError("Invalid or incomplete original twelve-run P1 ZIP")
            for run in manifest["runs"]:
                for rel, record in run["files"].items():
                    if rel not in names or hashlib.sha256(archive.read(rel)).hexdigest() != record["sha256"]:
                        raise ValueError(f"P1 ZIP hash mismatch: {rel}")
    checksums = {rel: {"sha256": sha(source), "bytes": source.stat().st_size}
                 for rel, source in inputs}
    metadata = {
        "scope": "review handoff; not yet journal-submission-ready",
        "source_files": checksums,
        "original_p1_evidence": {"included": bool(p1), "sha256": sha(p1) if p1 else None},
        "external_or_unperformed": [
            "P4 original paired outputs and LoRA adapter",
            "P3 two-human independent annotation",
            "P4 blinded human hint ratings",
            "P2 independent cross-family training seeds"
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(output, "x", compression=ZIP_DEFLATED) as z:
            for rel, source in inputs:
                z.write(source, rel)
            if p1:
                z.write(p1, "evidence/p1_v2_evidence.zip")
            z.writestr("HANDOFF_README.md", NOTICE)
            z.writestr("RELEASE_MANIFEST.json", json.dumps(metadata, indent=2) + "\n")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(f"Created review bundle: {output}")
    print(f"Bundle SHA-256: {sha(output)}")
    return metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--p1-archive", type=Path)
    args = p.parse_args()
    package(args.root, args.out, p1_archive=args.p1_archive)

if __name__ == "__main__":
    main()

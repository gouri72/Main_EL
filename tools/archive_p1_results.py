#!/usr/bin/env python3
"""Read-only verification and archival of original P1 v2 raw experiment outputs.

Usage:
  python tools/archive_p1_results.py --out artifacts/p1_v2_evidence.zip

The command does NOT rerun experiments, modify source results, or overwrite an
existing archive. It checks each configuration, seed/phase/episode/step count,
and CSV provenance before saving a ZIP with a SHA-256 manifest.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "always_full", "always_none", "fixed_human_first", "rule_adaptive",
    "tabular_q", "linear_q", "linucb", "opblind_state", "opblind_caps",
    "opblind_all", "performance_only", "loo_transfer",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_csv(path, exp_id, cfg_hash, seeds, phases, steps_per_episode=None):
    counts = Counter()
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        header = f.readline().strip()
        if f"config_hash={cfg_hash}" not in header or f"exp_id={exp_id}" not in header:
            raise ValueError(f"{path}: config hash or experiment ID missing/mismatched")
        reader = csv.DictReader(f)
        required = {"seed", "phase", "episode"}
        if steps_per_episode is not None:
            required.add("task_idx")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required CSV columns {required}")
        for row in reader:
            try:
                seed = int(row["seed"])
                phase = row["phase"]
                episode = int(row["episode"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid seed, phase or episode") from exc
            if seed not in seeds or phase not in phases:
                raise ValueError(f"{path}: unexpected seed/phase {seed}/{phase}")
            if not 0 <= episode < phases[phase]:
                raise ValueError(f"{path}: invalid episode {phase}/{seed}/{episode}")
            identity = (seed, phase, episode)
            if steps_per_episode is not None:
                try:
                    idx = int(row["task_idx"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: invalid task index") from exc
                if not 0 <= idx < steps_per_episode:
                    raise ValueError(f"{path}: task index out of range: {idx}")
                identity += (idx,)
            if identity in seen:
                raise ValueError(f"{path}: duplicate row identity {identity}")
            seen.add(identity)
            counts[(seed, phase)] += 1
    multiplier = steps_per_episode or 1
    expected = {(seed, phase): n * multiplier for seed in seeds for phase, n in phases.items()}
    if counts != Counter(expected):
        raise ValueError(f"{path}: incomplete or excess rows; got {dict(counts)}, expected {expected}")
    return sum(counts.values())


def validate_run(root, name):
    exp_id = f"A_novice_v2__{name}"
    cfg_path = root / "configs" / "grid" / f"{exp_id}.yaml"
    folder = root / "results" / exp_id
    source = folder / "config.yaml"
    episodes = folder / "episodes.csv"
    steps = folder / "steps.csv"
    for path in (cfg_path, source, episodes, steps):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required original experiment file missing/unsafe: {path}")

    if cfg_path.read_bytes() != source.read_bytes():
        raise ValueError(f"{exp_id}: archived config differs from committed config")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if config["exp_id"] != exp_id:
        raise ValueError(f"{exp_id}: configuration exp_id mismatch")
    seeds = tuple(int(s) for s in config["seeds"])
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError(f"{exp_id}: missing or repeated seeds")
    phases = {"train": int(config["n_episodes"]), "eval": int(config["n_eval_episodes"])}
    if phases["eval"] <= 0 or phases["train"] < 0:
        raise ValueError(f"{exp_id}: invalid phase counts")
    tasks = int(config["data"]["session_tasks"])
    if tasks <= 0:
        raise ValueError(f"{exp_id}: invalid task count")
    cfg_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    episode_count = check_csv(episodes, exp_id, cfg_hash, seeds, phases)
    step_count = check_csv(steps, exp_id, cfg_hash, seeds, phases, tasks)
    return {
        "exp_id": exp_id, "config_hash": cfg_hash, "seeds": list(seeds),
        "train_episodes_per_seed": phases["train"], "eval_episodes_per_seed": phases["eval"],
        "steps_per_episode": tasks, "episode_rows": episode_count,
        "step_rows": step_count,
        "files": {str(p.relative_to(root)).replace("\\", "/"): {
            "sha256": sha256(p), "bytes": p.stat().st_size
        } for p in (cfg_path, source, episodes, steps)}
    }


def archive(root, out, run_names=RUNS):
    root = Path(root).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Archive already exists; refusing to overwrite: {out}")
    if len(set(run_names)) != len(run_names):
        raise ValueError("duplicate experiment names")
    if any(name not in RUNS for name in run_names):
        raise ValueError("unknown experiment name")
    reports = [validate_run(root, name) for name in run_names]
    manifest = {
        "format_version": 1,
        "scope": "Original local P1 v2 outputs; not reconstructed from summary tables",
        "runs": reports,
    }
    # Check all inputs before creating an archive. Remove partial archive on error.
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(out, "x", compression=ZIP_DEFLATED, compresslevel=6) as z:
            for report in reports:
                for rel in report["files"]:
                    z.write(root / rel, arcname=rel)
            z.writestr("P1_EVIDENCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    except Exception:
        out.unlink(missing_ok=True)
        raise
    print(f"Verified and archived {len(reports)} original v2 experiments -> {out}")
    print("Archive SHA-256:", sha256(out))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    ap.add_argument("--out", type=Path, required=True, help="New ZIP path, must not already exist")
    args = ap.parse_args()
    archive(args.root, args.out)


if __name__ == "__main__":
    main()

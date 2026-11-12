#!/usr/bin/env python3
"""Generate the experiment grid (one YAML per experiment, configs/grid/) and run it in parallel.

  python run_grid.py --generate-only
  python run_grid.py --workers 4 --seeds 10 --episodes 200 --families A B C --profiles novice hint-dependent intermediate
exp_id = <family>_<profile>__<variant>; results go to results/<exp_id>/. Finished experiments are skipped (use --redo to force).
Variants: 10 policies/baselines + leave-one-out reward ablations + operation-blind ablations (see rl/configured_env.py).
"""
import argparse, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
POLICIES = ["tabular_q", "performance_only", "domain_specific", "linear_q", "linucb",
            "always_full", "always_none", "fixed_human_first", "fixed_ladder", "rule_adaptive"]
ABLATIONS = {
    "loo_progress": ["progress"], "loo_transfer": ["transfer"],
    "loo_full_assist": ["full_assist_penalty"], "loo_takeover": ["takeover_penalty"],
    "opblind_all": ["operation_state", "operation_penalty", "operation_caps"],
    "opblind_state": ["operation_state"], "opblind_penalty": ["operation_penalty"], "opblind_caps": ["operation_caps"],
}


def make_cfg(family, profile, variant, policy, ablate, seeds, episodes):
    return {
        "exp_id": f"{family}_{profile}__{variant}",
        "seeds": list(range(seeds)),
        "n_episodes": episodes,
        "data": {"tasks": "data/tasks.jsonl", "hints": "data/hints.jsonl", "session_tasks": 8},
        "simulator": {"family": family, "profile": profile},
        "reward": {"w_progress": 1.0, "w_transfer": 2.0, "w_full_assist_penalty": 0.5,
                   "w_takeover_penalty": 1.0, "ablate": ablate},
        "policy": {"name": policy},
    }


def run_one(path):
    r = subprocess.run([sys.executable, str(ROOT / "run_experiment.py"), str(path)], capture_output=True, text=True)
    return path.stem, r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--profiles", nargs="+", default=["novice", "hint-dependent", "intermediate"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    out = ROOT / "configs" / "grid"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fam in a.families:
        for prof in a.profiles:
            variants = [(p, p, []) for p in POLICIES] + [(v, "tabular_q", ab) for v, ab in ABLATIONS.items()]
            for variant, policy, ablate in variants:
                cfg = make_cfg(fam, prof, variant, policy, ablate, a.seeds, a.episodes)
                p = out / f"{cfg['exp_id']}.yaml"
                p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                paths.append(p)
    print(f"{len(paths)} configs in {out}")
    if a.generate_only:
        return
    todo = [p for p in paths if a.redo or not (ROOT / "results" / p.stem / "episodes.csv").exists()]
    print(f"running {len(todo)} experiments ({len(paths) - len(todo)} already done) with {a.workers} workers")
    failed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (name, code, last) in enumerate(ex.map(run_one, todo), 1):
            print(f"[{i}/{len(todo)}] {name}: {'ok' if code == 0 else 'FAILED'} {last}", flush=True)
            failed += code != 0
    print("done" if not failed else f"{failed} experiment(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
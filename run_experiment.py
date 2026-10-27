#!/usr/bin/env python3
"""Experiment runner (P4). One YAML config -> results/<exp_id>/{steps,episodes}.csv + config.yaml.
  python run_experiment.py configs/example.yaml
Works against P1's current stub env/policy. As P1 adds richer info keys (see STEP_INFO/EP_INFO), they are logged
automatically; missing keys are written as empty (NaN in pandas). Read results with pd.read_csv(path, comment='#')."""
import argparse, hashlib, importlib, json, random, shutil, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STEPS_COLS = ["exp_id", "config_hash", "policy", "sim_family", "profile", "seed", "episode", "task_idx",
              "action", "route", "level", "reward", "success", "domain", "op", "time", "frustrated", "abandoned", "copied"]
EPS_COLS = ["exp_id", "config_hash", "policy", "sim_family", "profile", "seed", "episode", "total_reward",
            "pre_transfer", "post_transfer", "transfer_gain", "n_steps", "full_assist_rate", "mean_level",
            "first_attempt_rate", "assist_spent", "takeover_rate"]

def load_cfg(path):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    body = json.dumps(cfg, sort_keys=True)
    cfg["_hash"] = hashlib.sha256(body.encode()).hexdigest()[:8]
    return cfg

def set_seed(s):
    random.seed(s); np.random.seed(s)

def load_tasks(path):
    return [json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip()]

def build(cfg, seed, tasks):
    """Resolve env + policy. P1: extend this registry when you add policies/baselines."""
    from rl.env import HumanFirstEnv, FAKE_TASKS
    from rl.policy import TabularQPolicy
    base = [t for t in tasks if not t["task_id"].endswith("_T")]
    env_tasks = base[: cfg["data"].get("session_tasks", 8)] if base else FAKE_TASKS
    env = HumanFirstEnv(env_tasks)
    pc = cfg["policy"]
    if pc["name"] == "tabular_q":
        pol = TabularQPolicy(alpha=pc.get("alpha", 0.1), gamma=pc.get("gamma", 0.9),
                             epsilon=pc.get("epsilon", 0.1), seed=seed)
    else:
        raise NotImplementedError(f"policy '{pc['name']}' not registered yet (P1: add it in run_experiment.build)")
    return env, pol, env_tasks

def write_csv(path, cfg, cols, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# config_hash={cfg['_hash']} exp_id={cfg['exp_id']}\n")
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); a = ap.parse_args()
    cfg = load_cfg(a.config)
    tasks = load_tasks(cfg["data"]["tasks"])
    meta = dict(exp_id=cfg["exp_id"], config_hash=cfg["_hash"], policy=cfg["policy"]["name"],
                sim_family=cfg["simulator"]["family"], profile=cfg["simulator"]["profile"])
    steps, eps = [], []
    for seed in cfg["seeds"]:
        set_seed(seed)
        env, pol, env_tasks = build(cfg, seed, tasks)
        for ep in range(cfg["n_episodes"]):
            state, done, tot, k = env.reset(), False, 0.0, 0
            lv_hist, full, first = [], 0, 0
            info = {}
            while not done:
                act = int(pol.get_action(state, is_training=True))
                nxt, r, done, info = env.step(act)
                pol.update(state, act, r, nxt, done)
                route, level = act // 5, act % 5
                t = env_tasks[min(k, len(env_tasks) - 1)]
                steps.append({**meta, "seed": seed, "episode": ep, "task_idx": k, "action": act, "route": route,
                              "level": level, "reward": round(r, 4), "success": info.get("immediate_success"),
                              "domain": t.get("domain"), "op": t["operations"][0]["op"],
                              "time": info.get("time"), "frustrated": info.get("frustrated"),
                              "abandoned": info.get("abandoned"), "copied": info.get("copied")})
                lv_hist.append(level); full += level == 4; first += route == 0
                tot += r; k += 1; state = nxt
            pre, post = info.get("pre_transfer"), info.get("transfer_success")
            if post is not None: post = int(post)
            gain = None if (pre is None or post is None or pre >= 1) else round((post - pre) / (1 - pre), 4)
            eps.append({**meta, "seed": seed, "episode": ep, "total_reward": round(tot, 4), "pre_transfer": pre,
                        "post_transfer": post, "transfer_gain": gain, "n_steps": k,
                        "full_assist_rate": round(full / k, 4), "mean_level": round(float(np.mean(lv_hist)), 4),
                        "first_attempt_rate": round(first / k, 4), "assist_spent": int(sum(lv_hist)),
                        "takeover_rate": info.get("takeover_rate")})
    out = ROOT / "results" / cfg["exp_id"]; out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "steps.csv", cfg, STEPS_COLS, steps)
    write_csv(out / "episodes.csv", cfg, EPS_COLS, eps)
    shutil.copy(a.config, out / "config.yaml")
    print(f"wrote {len(steps)} step rows, {len(eps)} episode rows -> {out}  (config_hash={cfg['_hash']})")

if __name__ == "__main__":
    main()
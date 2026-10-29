#!/usr/bin/env python3
"""Experiment runner (P4). One YAML config -> results/<exp_id>/{steps,episodes}.csv + config.yaml.
  python run_experiment.py configs/example.yaml
Each episode = a session of `data.session_tasks` tasks drawn at random (seeded) from P3's session pool, followed by the
unaided transfer block (the _T variants of those tasks). If rl.env.HumanFirstEnv accepts `transfer_tasks=`, they are passed
to it; otherwise the env keeps its own transfer logic. Info keys read from env.step (all optional, empty if missing):
  immediate_success, time, frustrated, abandoned, copied, op, actual_level, pre_transfer, transfer_success, takeover_rate.
Read results with pandas.read_csv(path, comment='#')."""
import argparse, hashlib, inspect, json, random, shutil, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.loader import load_tasks, session_pool, transfer_of

STEPS_COLS = ["exp_id", "config_hash", "policy", "sim_family", "profile", "seed", "episode", "task_idx", "task_id",
              "action", "route", "requested_level", "level", "reward", "success", "domain", "op",
              "time", "frustrated", "abandoned", "copied"]
EPS_COLS = ["exp_id", "config_hash", "policy", "sim_family", "profile", "seed", "episode", "task_ids", "total_reward",
            "pre_transfer", "post_transfer", "transfer_gain", "n_steps", "full_assist_rate", "mean_level",
            "first_attempt_rate", "assist_spent", "takeover_rate"]

def load_cfg(path):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    cfg["_hash"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return cfg

def set_seed(s):
    random.seed(s); np.random.seed(s)

def build_policy(cfg, seed):
    """P1: register new policies/baselines here."""
    from rl.policy import TabularQPolicy
    pc = cfg["policy"]
    if pc["name"] == "tabular_q":
        return TabularQPolicy(alpha=pc.get("alpha", 0.1), gamma=pc.get("gamma", 0.9),
                              epsilon=pc.get("epsilon", 0.1), seed=seed)
    raise NotImplementedError(f"policy '{pc['name']}' not registered yet (P1: add it in run_experiment.build_policy)")

def make_env(session, transfer):
    from rl.env import HumanFirstEnv
    if "transfer_tasks" in inspect.signature(HumanFirstEnv.__init__).parameters:
        return HumanFirstEnv(session, transfer_tasks=transfer)
    return HumanFirstEnv(session)

def write_csv(path, cfg, cols, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# config_hash={cfg['_hash']} exp_id={cfg['exp_id']}\n")
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); a = ap.parse_args()
    cfg = load_cfg(a.config)
    all_tasks = load_tasks(ROOT / cfg["data"]["tasks"])
    pool = session_pool(all_tasks, cfg["data"].get("domain"))
    n_sess = min(cfg["data"].get("session_tasks", 8), len(pool))
    meta = dict(exp_id=cfg["exp_id"], config_hash=cfg["_hash"], policy=cfg["policy"]["name"],
                sim_family=cfg["simulator"]["family"], profile=cfg["simulator"]["profile"])
    steps, eps = [], []
    for seed in cfg["seeds"]:
        set_seed(seed)
        rng = random.Random(seed)                   # task sampling: reproducible per seed
        pol = build_policy(cfg, seed)
        for ep in range(cfg["n_episodes"]):
            session = rng.sample(pool, n_sess)
            transfer = [transfer_of(all_tasks, t) for t in session]
            env = make_env(session, transfer)
            state, done, tot, k = env.reset(), False, 0.0, 0
            lv_hist, full, first, info = [], 0, 0, {}
            while not done:
                act = int(pol.get_action(state, is_training=True))
                nxt, r, done, info = env.step(act)
                pol.update(state, act, r, nxt, done)
                route, req = act // 5, act % 5
                level = info.get("actual_level", req)        # level after operation caps, if the env reports it
                t = session[min(k, len(session) - 1)]
                steps.append({**meta, "seed": seed, "episode": ep, "task_idx": k, "task_id": t["task_id"],
                              "action": act, "route": route, "requested_level": req, "level": level,
                              "reward": round(r, 4), "success": info.get("immediate_success"),
                              "domain": t.get("domain"), "op": info.get("op", t["operations"][0]["op"]),
                              "time": info.get("time"), "frustrated": info.get("frustrated"),
                              "abandoned": info.get("abandoned"), "copied": info.get("copied")})
                lv_hist.append(level); full += level == 4; first += route == 0
                tot += r; k += 1; state = nxt
            pre, post = info.get("pre_transfer"), info.get("transfer_success")
            pre = None if pre is None else float(pre); post = None if post is None else float(post)
            gain = None if (pre is None or post is None or pre >= 1) else round((post - pre) / (1 - pre), 4)
            eps.append({**meta, "seed": seed, "episode": ep, "task_ids": "|".join(t["task_id"] for t in session),
                        "total_reward": round(tot, 4), "pre_transfer": pre, "post_transfer": post,
                        "transfer_gain": gain, "n_steps": k, "full_assist_rate": round(full / k, 4),
                        "mean_level": round(float(np.mean(lv_hist)), 4), "first_attempt_rate": round(first / k, 4),
                        "assist_spent": int(sum(lv_hist)), "takeover_rate": info.get("takeover_rate")})
    out = ROOT / "results" / cfg["exp_id"]; out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "steps.csv", cfg, STEPS_COLS, steps)
    write_csv(out / "episodes.csv", cfg, EPS_COLS, eps)
    shutil.copy(a.config, out / "config.yaml")
    print(f"wrote {len(steps)} step rows, {len(eps)} episode rows -> {out}  (config_hash={cfg['_hash']})")

if __name__ == "__main__":
    main()
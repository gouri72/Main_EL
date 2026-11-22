import csv
from pathlib import Path

import numpy as np

from data.loader import load_tasks, session_pool, draw_session
from rl.configured_env import ConfiguredEnv
from rl.policy import TabularQPolicy


FAMILIES = ("A", "B", "C")
PROFILE = "novice"
N_TRAIN_EPISODES_PER_FAMILY = 120
N_EVAL_EPISODES = 50
SESSION_TASKS = 8
TRANSFER_REPEATS = 10

OUTPUT = Path("results/person2_cross_family.csv")


def make_env(family, seed, all_tasks, pool):
    return ConfiguredEnv(
        tasks=pool[:SESSION_TASKS],
        profile=PROFILE,
        family=family,
        seed=seed,
        all_tasks=all_tasks,
        reward={
            "w_progress": 1.0,
            "w_transfer": 2.0,
            "w_full_assist_penalty": 0.5,
            "w_takeover_penalty": 1.0,
        },
        discretize=True,
        transfer_repeats=TRANSFER_REPEATS,
    )


def run_episode(env, policy, pool, all_tasks, episode_seed, training):
    rng = np.random.default_rng(episode_seed)

    tasks = draw_session(
        pool=pool,
        rng=rng,
        n=SESSION_TASKS,
    )

    env.tasks = tasks
    env.all_tasks = all_tasks
    env.seed = episode_seed

    state = env.reset()
    done = False
    total_reward = 0.0
    successes = 0
    total_time = 0.0
    levels = []
    human_first = 0
    frustration = 0
    abandoned = 0
    copied = 0
    info = {}

    while not done:
        valid_actions = env.get_valid_actions()

        action = policy.get_action(
            state,
            valid_actions=valid_actions,
            is_training=training,
        )

        next_state, reward, done, info = env.step(action)

        next_valid = (
            env.get_valid_actions() if not done else []
        )

        if training:
            policy.update(
                state,
                action,
                reward,
                next_state,
                done,
                next_valid,
            )

        successes += int(info["immediate_success"])
        total_reward += float(reward)
        total_time += float(info.get("time", 0))
        levels.append(int(info.get("actual_level", 0)))
        human_first += int(info.get("route", action // 5) == 0)
        frustration += int(info.get("frustrated", False))
        abandoned += int(info.get("abandoned", False))
        copied += int(info.get("copied", False))

        state = next_state

    return {
        "reward": total_reward,
        "practice_success": successes / SESSION_TASKS,
        "transfer": info.get("transfer_success"),
        "pre_transfer": info.get("pre_transfer"),
        "mean_time": total_time / SESSION_TASKS,
        "mean_level": float(np.mean(levels)),
        "human_first_rate": human_first / SESSION_TASKS,
        "frustration_rate": frustration / SESSION_TASKS,
        "abandonment_rate": abandoned / SESSION_TASKS,
        "copying_rate": copied / SESSION_TASKS,
    }


def main():
    all_tasks = load_tasks("data/tasks.jsonl")
    pool = session_pool(all_tasks)

    rows = []

    for heldout in FAMILIES:
        training_families = [
            family for family in FAMILIES
            if family != heldout
        ]

        print(
            f"\nFold: test {heldout} | "
            f"train {training_families}"
        )

        policy = TabularQPolicy(
            alpha=0.1,
            gamma=0.95,
            epsilon=0.1,
            seed=100 + ord(heldout),
        )

        # Train on both available families using one shared policy.
        for family in training_families:
            env = make_env(family, 100, all_tasks, pool)

            for episode in range(N_TRAIN_EPISODES_PER_FAMILY):
                episode_seed = (
                    10000
                    + ord(heldout) * 1000
                    + ord(family) * 100
                    + episode
                )

                run_episode(
                    env,
                    policy,
                    pool,
                    all_tasks,
                    episode_seed,
                    training=True,
                )

            print(
                f"  Training completed on Family {family}: "
                f"{N_TRAIN_EPISODES_PER_FAMILY} episodes"
            )

        # Freeze policy: no updates on the held-out family.
        env = make_env(heldout, 200, all_tasks, pool)

        for episode in range(N_EVAL_EPISODES):
            episode_seed = (
                500000
                + ord(heldout) * 1000
                + episode
            )

            metrics = run_episode(
                env,
                policy,
                pool,
                all_tasks,
                episode_seed,
                training=False,
            )

            rows.append({
                "heldout_family": heldout,
                "training_families": "+".join(training_families),
                "profile": PROFILE,
                "episode": episode,
                **metrics,
            })

        test_rows = [
            r for r in rows
            if r["heldout_family"] == heldout
        ]

        print(
            f"  Frozen evaluation on Family {heldout}: "
            f"transfer={np.mean([r['transfer'] for r in test_rows]):.3f} | "
            f"practice={np.mean([r['practice_success'] for r in test_rows]):.3f} | "
            f"reward={np.mean([r['reward'] for r in test_rows]):.3f}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCompleted {len(rows)} frozen evaluation episodes.")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

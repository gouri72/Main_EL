
import numpy as np
from collections import defaultdict


def random_argmax(values, rng):
    values = np.asarray(values, dtype=float)
    candidates = np.flatnonzero(
        np.isclose(values, np.max(values))
    )
    return int(rng.choice(candidates))


def get_features(state):
    """
    Normalize the eight observable features.
    No hidden simulator mastery/dependence is exposed.
    """
    scales = np.array(
        [1, 3, 2, 2, 3, 2, 3, 1],
        dtype=float
    )

    x = np.asarray(state, dtype=float) / scales
    return np.append(x, 1.0)


def valid_ids(valid_actions, num_actions):
    if valid_actions is None:
        return list(range(num_actions))

    actions = [int(a) for a in valid_actions]

    if not actions:
        raise ValueError("No valid actions available")

    if any(a < 0 or a >= num_actions for a in actions):
        raise ValueError("Invalid action ID")

    return actions


class TabularQPolicy:
    def __init__(
        self, num_actions=10, alpha=0.1,
        gamma=0.95, epsilon=0.1, seed=42
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.q_table = defaultdict(
            lambda: np.zeros(num_actions)
        )

    def get_action(
        self, state, valid_actions=None, is_training=True
    ):
        actions = valid_ids(
            valid_actions, self.num_actions
        )

        key = tuple(state)

        if (
            is_training
            and self.rng.random() < self.epsilon
        ):
            return int(self.rng.choice(actions))

        q = np.full(self.num_actions, -np.inf)
        q[actions] = self.q_table[key][actions]

        return random_argmax(q, self.rng)

    def update(
        self, state, action, reward,
        next_state, done, next_valid_actions=None
    ):
        key = tuple(state)
        next_key = tuple(next_state)

        if done:
            future = 0.0
        else:
            actions = valid_ids(
                next_valid_actions, self.num_actions
            )
            future = np.max(
                self.q_table[next_key][actions]
            )

        target = reward + self.gamma * future

        self.q_table[key][action] += self.alpha * (
            target - self.q_table[key][action]
        )


class LinearQPolicy:
    def __init__(
        self, num_actions=10, feature_dim=9,
        alpha=0.01, gamma=0.95,
        epsilon=0.1, seed=42
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.weights = np.zeros(
            (num_actions, feature_dim)
        )

    def get_action(
        self, state, valid_actions=None, is_training=True
    ):
        actions = valid_ids(
            valid_actions, self.num_actions
        )

        if (
            is_training
            and self.rng.random() < self.epsilon
        ):
            return int(self.rng.choice(actions))

        x = get_features(state)

        q = np.full(self.num_actions, -np.inf)
        q[actions] = (self.weights @ x)[actions]

        return random_argmax(q, self.rng)

    def update(
        self, state, action, reward,
        next_state, done, next_valid_actions=None
    ):
        x = get_features(state)

        if done:
            future = 0.0
        else:
            next_x = get_features(next_state)
            actions = valid_ids(
                next_valid_actions, self.num_actions
            )
            future = np.max(
                (self.weights @ next_x)[actions]
            )

        target = reward + self.gamma * future

        error = target - self.weights[action] @ x

        self.weights[action] += (
            self.alpha * error * x
        )


class LinUCBPolicy:
    """
    Contextual-bandit baseline.

    LinUCB learns from immediate contextual rewards;
    it is not a full temporal-difference RL algorithm.
    """

    def __init__(
        self, num_actions=10, feature_dim=9,
        alpha=1.0, seed=42
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        self.A = [
            np.eye(feature_dim)
            for _ in range(num_actions)
        ]
        self.b = [
            np.zeros(feature_dim)
            for _ in range(num_actions)
        ]

    def get_action(
        self, state, valid_actions=None, is_training=True
    ):
        actions = valid_ids(
            valid_actions, self.num_actions
        )

        x = get_features(state)

        scores = np.full(
            self.num_actions, -np.inf
        )

        for action in actions:
            inverse_x = np.linalg.solve(
                self.A[action], x
            )

            theta = np.linalg.solve(
                self.A[action], self.b[action]
            )

            score = theta @ x

            if is_training:
                uncertainty = np.sqrt(
                    max(0.0, x @ inverse_x)
                )
                score += self.alpha * uncertainty

            scores[action] = score

        return random_argmax(scores, self.rng)

    def update(
        self, state, action, reward,
        next_state, done, next_valid_actions=None
    ):
        x = get_features(state)

        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

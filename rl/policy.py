
import numpy as np
from collections import defaultdict


def random_argmax(values, rng):
    """Randomly choose among actions tied for the highest value."""
    values = np.asarray(values)
    candidates = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(candidates))


def get_features(state):
    """Convert the 8-variable state into a 9-dimensional feature vector."""
    return np.asarray(list(state) + [1.0], dtype=np.float32)


class TabularQPolicy:
    def __init__(self, num_actions=10, alpha=0.1, gamma=0.9,
                 epsilon=0.1, seed=42):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = defaultdict(lambda: np.zeros(num_actions))

    def get_action(self, state, is_training=True):
        state_key = tuple(state)

        if is_training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))

        return random_argmax(self.q_table[state_key], self.rng)

    def update(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        best_next_q = (
            np.max(self.q_table[next_state_key]) if not done else 0.0
        )
        target = reward + self.gamma * best_next_q
        error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * error


class LinearQPolicy:
    def __init__(self, num_actions=10, feature_dim=9, alpha=0.01,
                 gamma=0.9, epsilon=0.1, seed=42):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((num_actions, feature_dim))

    def get_action(self, state, is_training=True):
        features = get_features(state)

        if is_training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))

        return random_argmax(self.weights @ features, self.rng)

    def update(self, state, action, reward, next_state, done):
        features = get_features(state)
        next_features = get_features(next_state)

        best_next_q = (
            np.max(self.weights @ next_features) if not done else 0.0
        )
        target = reward + self.gamma * best_next_q
        current = self.weights[action] @ features
        error = target - current
        self.weights[action] += self.alpha * error * features


class LinUCBPolicy:
    def __init__(self, num_actions=10, feature_dim=9,
                 alpha=1.0, seed=42):
        self.num_actions = num_actions
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.A = [np.eye(feature_dim) for _ in range(num_actions)]
        self.b = [np.zeros(feature_dim) for _ in range(num_actions)]

    def get_action(self, state, is_training=True):
        features = get_features(state)
        scores = np.zeros(self.num_actions)

        for action in range(self.num_actions):
            inverse = np.linalg.solve(
                self.A[action], np.eye(len(features))
            )
            theta = inverse @ self.b[action]
            score = theta @ features

            if is_training:
                uncertainty = np.sqrt(
                    max(0.0, features @ inverse @ features)
                )
                score += self.alpha * uncertainty

            scores[action] = score

        return random_argmax(scores, self.rng)

    def update(self, state, action, reward, next_state, done):
        features = get_features(state)
        self.A[action] += np.outer(features, features)
        self.b[action] += reward * features

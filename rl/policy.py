# rl/policy.py
import numpy as np
from collections import defaultdict

class TabularQPolicy:
    def __init__(self, num_actions=10, alpha=0.1, gamma=0.9, epsilon=0.1, seed=42):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        
        # State will be converted to a tuple to act as a dictionary key
        self.q_table = defaultdict(lambda: np.zeros(num_actions))

    def get_action(self, state, is_training=True):
        state_key = tuple(state)
        # Epsilon-greedy exploration
        if is_training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.num_actions)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # Q-learning update rule
        best_next_q = np.max(self.q_table[next_state_key]) if not done else 0.0
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state_key][action]
        
        self.q_table[state_key][action] += self.alpha * td_error

def get_features(state):
    """Converts the 8-variable discrete state into a feature vector with a bias term."""
    return np.array(state + [1.0], dtype=np.float32)


class LinearQPolicy:
    def __init__(self, num_actions=10, feature_dim=9, alpha=0.01, gamma=0.9, epsilon=0.1, seed=42):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        # Weight matrix: one weight vector per action
        self.weights = np.zeros((num_actions, feature_dim))

    def get_action(self, state, is_training=True):
        features = get_features(state)
        if is_training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.num_actions)
        q_values = self.weights @ features
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done):
        features = get_features(state)
        next_features = get_features(next_state)
        
        best_next_q = np.max(self.weights @ next_features) if not done else 0.0
        td_target = reward + self.gamma * best_next_q
        current_q = self.weights[action] @ features
        
        td_error = td_target - current_q
        self.weights[action] += self.alpha * td_error * features

class LinUCBPolicy:
    def __init__(self, num_actions=10, feature_dim=9, alpha=1.0, seed=42):
        self.num_actions = num_actions
        self.alpha = alpha  # Exploration multiplier
        # Contextual bandit parameters per action
        self.A = [np.eye(feature_dim) for _ in range(num_actions)]
        self.b = [np.zeros(feature_dim) for _ in range(num_actions)]

    def get_action(self, state, is_training=True):
        features = get_features(state)
        
        if not is_training:
            # Greedy action: theta @ x
            p = [np.linalg.solve(self.A[a], self.b[a]) @ features for a in range(self.num_actions)]
            return np.argmax(p)
            
        p = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            variance = features.T @ A_inv @ features
            # Upper Confidence Bound
            p[a] = (theta @ features) + self.alpha * np.sqrt(variance)
        return np.argmax(p)

    def update(self, state, action, reward, next_state, done):
        # LinUCB operates as a contextual bandit, ignoring next_state and gamma
        features = get_features(state)
        self.A[action] += np.outer(features, features)
        self.b[action] += reward * features
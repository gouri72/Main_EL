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
# rl/baselines.py

class AlwaysFullPolicy:
    """Always gives full assistance immediately (AI_FIRST, level 4)."""
    def get_action(self, state, is_training=False):
        # route 1 (AI_FIRST) * 5 + level 4 = action_id 9
        return 9 

class AlwaysNonePolicy:
    """Never gives assistance (HUMAN_FIRST, level 0)."""
    def get_action(self, state, is_training=False):
        # route 0 (HUMAN_FIRST) * 5 + level 0 = action_id 0
        return 0

class FixedLadderPolicy:
    """Progressively increases help if attempts increase (assumes attempt count is state index 5)."""
    def get_action(self, state, is_training=False):
        attempts = state[5] 
        # Cap at level 4. Route 0 (HUMAN_FIRST)
        level = min(attempts, 4)
        return 0 * 5 + level
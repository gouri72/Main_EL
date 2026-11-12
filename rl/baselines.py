# rl/baselines.py

class AlwaysFullPolicy:
    """Always gives full assistance immediately (AI_FIRST, max level)."""
    def get_action(self, state, valid_actions=None, is_training=False):
        action_id = 9 # AI_FIRST, Level 4
        if valid_actions is not None and action_id not in valid_actions:
            return max(valid_actions) # Fallback to highest allowed cap
        return action_id

class AlwaysNonePolicy:
    """Never gives assistance (HUMAN_FIRST, level 0)."""
    def get_action(self, state, valid_actions=None, is_training=False):
        action_id = 0 # HUMAN_FIRST, Level 0
        if valid_actions is not None and action_id not in valid_actions:
            return min(valid_actions)
        return action_id

class FixedLadderPolicy:
    """Progressively increases help if attempts increase (assumes attempt count is state index 5)."""
    def get_action(self, state, valid_actions=None, is_training=False):
        attempts = state[5] 
        level = min(attempts, 4)
        action_id = 0 * 5 + level # HUMAN_FIRST
        
        if valid_actions is not None and action_id not in valid_actions:
            valid_for_route = [a for a in valid_actions if a // 5 == 0]
            return max(valid_for_route) if valid_for_route else valid_actions[0]
        return action_id

class FixedHumanFirstPolicy:
    """Always forces an independent attempt, followed by max assistance (HUMAN_FIRST, max level)."""
    def get_action(self, state, valid_actions=None, is_training=False):
        action_id = 4 # HUMAN_FIRST, Level 4
        if valid_actions is not None and action_id not in valid_actions:
            valid_for_route = [a for a in valid_actions if a // 5 == 0]
            return max(valid_for_route) if valid_for_route else valid_actions[0]
        return action_id

class RuleBasedPolicy:
    """Strong heuristic baseline that adapts to learner state and task difficulty."""
    def get_action(self, state, valid_actions=None, is_training=False):
        domain, op, dbin, obin, diff, att, hlp, frs = state
        
        # If frustrated or struggling (high attempts), intervene immediately (AI_FIRST, High Help)
        if frs > 0 or att >= 2:
            route = 1
            level = 4
        # If difficulty is high and operation knowledge is low, give AI_FIRST moderate help
        elif diff >= 2 and obin == 0:
            route = 1
            level = 3
        # If knowledge is high, let them try independently (HUMAN_FIRST, No Help)
        elif obin == 2:
            route = 0
            level = 0
        # Default fallback for intermediate states (HUMAN_FIRST, Moderate Help)
        else:
            route = 0
            level = 2
            
        action_id = route * 5 + level
        
        if valid_actions is not None and action_id not in valid_actions:
            valid_for_route = [a for a in valid_actions if a // 5 == route]
            if valid_for_route:
                # Pick the closest valid level for this route
                return min(valid_for_route, key=lambda a: abs((a % 5) - level))
            else:
                return valid_actions[-1]
                
        return action_id
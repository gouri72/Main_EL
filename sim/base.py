# sim/base.py
class Learner:
    def reset(self, profile, seed) -> None:
        self.profile = profile
        self.seed = seed

    def attempt(self, task, op, level, hint) -> dict:
        # Dummy behavior: success probability scales with assistance level
        success = level >= 2 
        return {"success": success, "time": 15.0, "frustrated": False, "abandoned": False, "copied": (level == 4)}

    def transfer(self, task) -> dict:
        # Dummy unaided transfer
        return {"success": True}
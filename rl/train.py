# rl/train.py
from rl.env import HumanFirstEnv, FAKE_TASKS
from rl.policy import TabularQPolicy
from rl.baselines import AlwaysNonePolicy, FixedLadderPolicy

def run_episode(env, policy, name="Policy"):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # FixedLadder needs is_training=False to avoid epsilon exploration
        action = policy.get_action(state, is_training=True if name == "Q-Learning" else False)
        next_state, reward, done, info = env.step(action)
        
        if hasattr(policy, 'update'):
            policy.update(state, action, reward, next_state, done)
            
        state = next_state
        total_reward += reward
        
    print(f"{name} | Total Reward: {total_reward:.2f} | Transfer Success: {info.get('transfer_success')}")

if __name__ == "__main__":
    env = HumanFirstEnv(FAKE_TASKS)
    
    print("--- Testing Baselines ---")
    run_episode(env, AlwaysNonePolicy(), "Always None")
    run_episode(env, FixedLadderPolicy(), "Fixed Ladder")
    
    print("\n--- Testing Tabular Q-Learning ---")
    q_policy = TabularQPolicy()
    for i in range(1, 4):
        run_episode(env, q_policy, "Q-Learning")
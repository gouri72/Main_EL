# rl/train.py
from rl.env import HumanFirstEnv
from rl.policy import TabularQPolicy
from rl.baselines import AlwaysNonePolicy, FixedLadderPolicy

def run_episode(env, policy, name="Policy"):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # FixedLadder needs is_training=False to avoid epsilon exploration
        action = policy.get_action(state, is_training=(name not in ["Always None", "Fixed Ladder"]))
        next_state, reward, done, info = env.step(action)
        
        if hasattr(policy, 'update'):
            policy.update(state, action, reward, next_state, done)
            
        state = next_state
        total_reward += reward
        
    print(f"{name} | Total Reward: {total_reward:.2f} | Transfer Success: {info.get('transfer_success')}")

if __name__ == "__main__":
    from rl.policy import LinearQPolicy, LinUCBPolicy

    env = HumanFirstEnv()
    print("--- Testing Baselines ---")
    run_episode(env, AlwaysNonePolicy(), "Always None")
    run_episode(env, FixedLadderPolicy(), "Fixed Ladder")
    
    print("\n--- Testing Tabular Q-Learning ---")
    q_policy = TabularQPolicy()
    for i in range(1, 4):
        run_episode(env, q_policy, "Tabular Q")

    print("\n--- Testing Linear Q-Learning ---")
    lin_q_policy = LinearQPolicy()
    for i in range(1, 4):
        run_episode(env, lin_q_policy, "Linear Q")
        
    print("\n--- Testing LinUCB ---")
    lin_ucb_policy = LinUCBPolicy()
    for i in range(1, 4):
        run_episode(env, lin_ucb_policy, "LinUCB")
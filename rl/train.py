# rl/train.py
from rl.env import HumanFirstEnv, FAKE_TASKS
from rl.policy import TabularQPolicy

def run_sanity_check():
    print("Initializing environment and policy...")
    env = HumanFirstEnv(FAKE_TASKS)
    policy = TabularQPolicy()

    # Run 5 dummy episodes
    for episode in range(1, 6):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = policy.get_action(state, is_training=True)
            next_state, reward, done, info = env.step(action)
            
            policy.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
        print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Transfer Success: {info.get('transfer_success')}")

if __name__ == "__main__":
    run_sanity_check()
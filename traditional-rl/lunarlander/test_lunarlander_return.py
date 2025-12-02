import argparse
import gymnasium as gym
import numpy as np
np.float_ = np.float64
import torch as th
from stable_baselines3 import PPO
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--policy_path', type=str, default="models_lunarlander/ppo_lunarlander_run.zip",
                    help="Path to the trained LunarLander policy.")
parser.add_argument('--num_episodes', type=int, default=100,
                    help="Number of evaluation episodes.")
parser.add_argument('--env_name', type=str, default="LunarLander-v2",
                    help="Environment name (default: LunarLander-v2).")
args = parser.parse_args()

def simulate_trajectory(policy, env):
    """
    Simulate one full episode in LunarLander-v2.
    """
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        # Predict action using the policy.
        action, _ = policy.predict(obs, deterministic=False)
        # Step the environment.
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
    return total_reward

def evaluate_policy(policy, env_name, num_episodes):
    """
    Evaluate the policy on the specified environment over a given number of episodes.
    """
    if env_name == 'FrozenLake-v1':
        from gymnasium.envs.toy_text.frozen_lake import generate_random_map
        env = gym.make(env_name, map_name='4x4', is_slippery=False)
    elif env_name == "highway-v0":
        import highway_env
        env = gym.make("highway-v0", render_mode=None, config={
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "vehicles_count": 10,
        })
    elif "merge" in env_name:
        import highway_env
        env = gym.make(env_name, render_mode=None)
    else:
        env = gym.make(env_name)
        
    returns = []
    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        ret = simulate_trajectory(policy, env)
        returns.append(ret)
    env.close()
    return np.mean(returns), np.std(returns), returns

if __name__ == "__main__":
    # Load the trained policy.
    policy = PPO.load(args.policy_path, device="cpu")
    print("Loaded policy from:", args.policy_path)
    
    # Evaluate the policy.
    avg_return, std_returns, returns = evaluate_policy(policy, args.env_name, args.num_episodes)
    
    print("Average return over {} episodes: {:.2f}".format(args.num_episodes, avg_return))
    print("Return standard deviation: {:.2f}".format(std_returns))
    print("Returns for each episode:", returns)

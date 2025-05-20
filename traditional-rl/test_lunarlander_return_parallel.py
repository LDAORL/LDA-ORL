import argparse
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--policy_path', type=str, default="models_lunarlander/ppo_lunarlander_run.zip",
                    help="Path to the trained LunarLander policy.")
parser.add_argument('--num_episodes', type=int, default=100,
                    help="Total number of evaluation episodes.")
parser.add_argument('--env_name', type=str, default="LunarLander-v2",
                    help="Environment name (default: LunarLander-v2).")
parser.add_argument('--num_envs', type=int, default=8,
                    help="Number of parallel environments for evaluation.")
args = parser.parse_args()

def evaluate_policy_vectorized(policy, env_name, num_episodes, num_envs):
    # Create a vectorized environment with the desired number of parallel envs.
    def make_env():
        return gym.make(env_name)
    env = DummyVecEnv([make_env for _ in range(num_envs)])

    # Reset all environments (DummyVecEnv returns only observations).
    obs = env.reset()
    episode_returns = np.zeros(num_envs)
    completed_returns = []

    # Continue until we have collected the desired number of episodes.
    while len(completed_returns) < num_episodes:
        # Batch predict actions.
        actions, _ = policy.predict(obs, deterministic=True)
        obs, rewards, dones, truncs, infos = env.step(actions)
        # Update return per environment.
        episode_returns += rewards.flatten()
        # Check for done/truncated and record finished episode returns.
        for i, done in enumerate(dones):
            if done:
                completed_returns.append(episode_returns[i])
                episode_returns[i] = 0.0  # Reset return for that environment.

    env.close()
    completed_returns = np.array(completed_returns[:num_episodes])
    return np.mean(completed_returns), np.std(completed_returns), completed_returns

if __name__ == "__main__":
    # Load the trained policy.
    policy = PPO.load(args.policy_path, device="cpu")
    print("Loaded policy from:", args.policy_path)
    
    avg_return, std_returns, returns = evaluate_policy_vectorized(
        policy, args.env_name, args.num_episodes, args.num_envs)
    
    print("Average return over {} episodes: {:.2f}".format(args.num_episodes, avg_return))
    print("Return standard deviation: {:.2f}".format(std_returns))
    print("Returns for each episode:", returns)

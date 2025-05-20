import argparse
import gymnasium as gym
import minigrid
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.wrappers import ImgObsWrapper
import numpy as np
import torch as th
from stable_baselines3 import PPO
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--policy_path', type=str, default="models_minigrid/minigrid_lr-5e-3_feat-64_step-80k/policy_grad_12800.zip")
parser.add_argument('--num_episodes', type=int, default=100)
parser.add_argument('--env_name', type=str, default="MiniGrid-Empty-5x5-v0")
args = parser.parse_args()

print('env is :', args.env_name)

def generate_fixed_initial_states(num_states=100, env_name="MiniGrid-Empty-5x5-v0"):
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)
    env = ImgObsWrapper(env)
    fixed_states = []
    for _ in range(num_states):
        obs, info = env.reset()
        # Optionally, if your environment supports saving its full internal state,
        # you could record that state. Here we just record the observation.
        fixed_states.append(obs.copy())
    env.close()
    return fixed_states

def simulate_trajectory(policy, env, initial_state):
    """
    Simulate one trajectory starting from initial_state.
    This example assumes that the environment has a set_state() method.
    If not, you need to modify your environment so that it can reset to a given state.
    """
    # Reset the environment (to initialize internal variables)
    obs, info = env.reset()
    # If your environment supports setting state, set it to the fixed initial state.
    if hasattr(env, "set_state"):
        env.set_state(initial_state)
        obs = initial_state
    else:
        # Otherwise, if you cannot manually set the state,
        # you could override the observation returned by reset.
        obs = initial_state
        
    # print('initial state', obs.shape)

    total_reward = 0.0
    done = False
    steps = 0

    # Gymnasium step returns: (obs, reward, terminated, truncated, info)
    while not done:
        # Use the policy to predict an action
        action, _ = policy.predict(obs, deterministic=False)
        # print(obs.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs.shape)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    return total_reward

def evaluate_policy_on_fixed_states(policy, fixed_states, env_name="MiniGrid-Empty-5x5-v0"):
    # Create a fresh environment for evaluation.
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)
    env = ImgObsWrapper(env)
    
    returns = []
    for init_state in tqdm(fixed_states):
        ret = simulate_trajectory(policy, env, init_state)
        returns.append(ret)
    env.close()

    return np.mean(returns), np.std(returns), returns

if __name__ == "__main__":
    # Example: Load a trained policy.
    policy_path = args.policy_path
    policy = PPO.load(policy_path, device="cpu", seed=42)
    print('loaded model with initial policy', policy_path)
    
    # Generate a fixed set of initial states.
    fixed_states = generate_fixed_initial_states(num_states=args.num_episodes, env_name=args.env_name)
    print("Generated", len(fixed_states), "fixed initial states.")
    
    # Evaluate the policy on these fixed initial states.
    avg_return, std_returns, returns = evaluate_policy_on_fixed_states(policy, fixed_states, env_name=args.env_name)
    print("Average return J over fixed states:", avg_return)
    print("Std return J over fixed states:", std_returns)
    print("All returns over fixed states:", returns)

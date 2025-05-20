import argparse
import os
import pickle
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import minigrid
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
import wandb

# =======================
# Custom CNN Feature Extractor
# =======================
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# Define a wrapper to restrict the action space
class RestrictedActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Redefine action space to only include Left (0), Right (1), Forward (2)
        self.action_space = gym.spaces.Discrete(3)  # 3 actions instead of 7
        # Mapping from new action indices to original MiniGrid actions
        self.action_mapping = {0: 0, 1: 1, 2: 2}  # 0: Left, 1: Right, 2: Forward

    def step(self, action):
        # Map the restricted action to the original action
        mapped_action = self.action_mapping[action]
        # Call the original environment's step function
        obs, reward, terminated, truncated, info = self.env.step(mapped_action)
        return obs, reward, terminated, truncated, info

# =======================
# Environment creation
# =======================
def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        env = RestrictedActionSpaceWrapper(env)  # restrict action space to Left, Right, Forward
        env = RecordEpisodeStatistics(env)  # for episode stats
        env = ImgObsWrapper(env)            # returns only the image observation (3,7,7)
        return env
    return _init

# =======================
# Callback to save states (observations) from rollouts
# =======================
class SaveStatesCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveStatesCallback, self).__init__(verbose)
        self.save_path = save_path
        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_advantages = []
        self.all_episode_starts = []
    
    def _on_step(self) -> bool:
        return True
    
    def on_rollout_end(self) -> None:
        if hasattr(self.model.rollout_buffer, "observations"):
            obs = self.model.rollout_buffer.observations.copy()
            self.all_states.extend(obs)
            
            actions = self.model.rollout_buffer.actions.copy()
            self.all_actions.extend(actions)
            
            rewards = self.model.rollout_buffer.rewards.copy()
            self.all_rewards.extend(rewards)
            
            advantages = self.model.rollout_buffer.advantages.copy()
            self.all_advantages.extend(advantages)
            
            episode_starts = self.model.rollout_buffer.episode_starts.copy()
            self.all_episode_starts.extend(episode_starts)
    
    def on_training_end(self) -> None:
        all_states_array = np.array(self.all_states)
        save_file = os.path.join(self.save_path, "all_states.npy")
        np.save(save_file, all_states_array)
        print(f"Saved all states to {save_file}")
        
        all_actions_array = np.array(self.all_actions)
        save_file = os.path.join(self.save_path, "all_actions.npy")
        np.save(save_file, all_actions_array)
        print(f"Saved all actions to {save_file}")
        
        all_rewards_array = np.array(self.all_rewards)
        save_file = os.path.join(self.save_path, "all_rewards.npy")
        np.save(save_file, all_rewards_array)
        print(f"Saved all rewards to {save_file}")
        
        all_advantages_array = np.array(self.all_advantages)
        save_file = os.path.join(self.save_path, "all_advantages.npy")
        np.save(save_file, all_advantages_array)
        print(f"Saved all advantages to {save_file}")
        
        all_episode_starts_array = np.array(self.all_episode_starts)
        save_file = os.path.join(self.save_path, "all_episode_starts.npy")
        np.save(save_file, all_episode_starts_array)
        print(f"Saved all episode starts to {save_file}")

# =======================
# Argument parsing
# =======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO on MiniGrid with custom CNN and data attribution logging."
    )
    # Training hyperparameters
    parser.add_argument("--total_timesteps", type=int, default=40960,
                        help="Total timesteps for training.")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--features_dim", type=int, default=256,
                        help="Feature dimension for the custom CNN extractor.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    
    # WandB settings.
    parser.add_argument("--project_name", type=str, default="minigrid-rl",
                        help="WandB project name.")
    parser.add_argument("--run_name", type=str, default="ppo_minigrid_run",
                        help="WandB run name.")
    parser.add_argument("--save_path", type=str, default="saved_models",
                        help="Directory to save models and mini-batches.")
    parser.add_argument('--optimizer_class', type=str, default="SGD")
    
    parser.add_argument('--env_name', type=str, default="MiniGrid-Empty-5x5-v0",
                        help="Name of the environment to train on.")
    return parser.parse_args()

# =======================
# Main training function
# =======================
def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize wandb for logging
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
    )
    config = wandb.config

    # Create environment wrapped in DummyVecEnv and VecMonitor
    env = DummyVecEnv([make_env(args.env_name)])
    env = VecMonitor(env)
    
    print('env:', args.env_name)
    print('number of actions:', env.action_space.n)
    
    # Define policy keyword arguments
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config.features_dim),
        optimizer_class=th.optim.SGD if args.optimizer_class == "SGD" else th.optim.Adam,
    )
    
    # Use "CnnPolicy" since we're using image observations
    # If you're not visualizing and the input is small, ensure you use your custom CNN extractor.
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda" if th.cuda.is_available() else "cpu",
        learning_rate=config.learning_rate,
        seed=config.seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tb_logs/"
    )
    
    # Save an initial copy of the policy for reference.
    model.save(os.path.join(args.save_path, "policy_grad_0.zip"))
    
    # Initialize update counter for saving batches and checkpoints.
    model._update_counter = 0
    
    # Patch the rollout buffer's get() method to save mini-batches as pickle files.
    original_get = model.rollout_buffer.get
    def patched_get(batch_size):
        for mini_batch in original_get(batch_size):
            model._update_counter += 1
            batch_filename = os.path.join(args.save_path, f"batch_{model._update_counter}.pkl")
            with open(batch_filename, "wb") as f:
                pickle.dump(mini_batch, f)
            yield mini_batch
    model.rollout_buffer.get = patched_get
    
    # Patch the optimizer's step method to save a policy checkpoint after each gradient update.
    original_optimizer_step = model.policy.optimizer.step
    def patched_optimizer_step(*opt_args, **opt_kwargs):
        result = original_optimizer_step(*opt_args, **opt_kwargs)
        checkpoint_filename = os.path.join(args.save_path, f"policy_grad_{model._update_counter}.zip")
        model.save(checkpoint_filename)
        return result
    model.policy.optimizer.step = patched_optimizer_step
    
    # Create the state-saving callback.
    save_states_callback = SaveStatesCallback(save_path=args.save_path, verbose=1)
    callback = CallbackList([WandbCallback(verbose=2), save_states_callback])
    
    # Begin training.
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback
    )
    
    # Save final model.
    final_model_path = os.path.join(args.save_path, "final_policy.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()

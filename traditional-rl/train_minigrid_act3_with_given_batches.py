import argparse
import os
import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.optim import SGD
from tqdm import tqdm

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import gymnasium as gym
import minigrid
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.wrappers import ImgObsWrapper

# -------------------------------
# Custom CNN extractor for image observations (input: (3,7,7))
# -------------------------------
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # expect 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1),  # (16, 5, 5)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),                # (32, 3, 3)
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

# -------------------------------
# Offline state-saving callback
# -------------------------------
class SaveStatesCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveStatesCallback, self).__init__(verbose)
        self.save_path = save_path
        self.all_states = []
    
    def _on_step(self) -> bool:
        return True
    
    def on_rollout_end(self) -> None:
        if hasattr(self.model.rollout_buffer, "observations"):
            obs = self.model.rollout_buffer.observations.copy()
            self.all_states.extend(obs)
    
    def on_training_end(self) -> None:
        all_states_array = np.array(self.all_states)
        save_file = os.path.join(self.save_path, "all_states.npy")
        np.save(save_file, all_states_array)
        print(f"Saved all states to {save_file}")

# -------------------------------
# Argument parsing
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline PPO training on MiniGrid with custom CNN, wandb logging, and batch checkpointing."
    )
    # Training hyperparameters.
    parser.add_argument("--total_timesteps", type=int, default=100_000,
                        help="Total timesteps for training (not used in offline loop).")
    parser.add_argument("--learning_rate", type=float, default=0.0003,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per rollout (not used in offline loop).")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="Entropy coefficient.")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="Clip range for PPO.")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epochs per update (not used in offline loop).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training.")
    
    # CNN extractor parameter.
    parser.add_argument("--features_dim", type=int, default=256,
                        help="Feature dimension for the custom CNN extractor.")

    # Environment hyperparameters.
    # For our current use case, we use MiniGrid-Empty-5x5.
    
    # Save path for models and states.
    parser.add_argument("--save_path", type=str, default="saved_models",
                        help="Directory to save model parameters and states.")
    
    # Offline batch training parameters.
    parser.add_argument("--batch_begin_idx", type=int, default=0,
                        help="Start index for training batches.")
    parser.add_argument("--n_batches", type=int, default=6400,
                        help="Number of batches to process.")
    parser.add_argument("--batches_dir", type=str, required=True,
                        help="Directory containing saved mini-batch pickle files.")
    parser.add_argument("--total_updates", type=int, default=6400,
                        help="Total number of gradient updates (for scheduling clip_range).")
    
    parser.add_argument("--normalize_advantage", action="store_true", default=False,
                        help="Whether to normalize advantages.")
    
    # WandB settings.
    parser.add_argument("--project_name", type=str, default="minigrid-rl",
                        help="WandB project name.")
    parser.add_argument("--run_name", type=str, default="ppo_minigrid_offline_run",
                        help="WandB run name.")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_policy", type=str, default='',
                        help="Path to an initial policy checkpoint.")
    
    parser.add_argument('--env_name', type=str, default="MiniGrid-Empty-5x5-v0")
    parser.add_argument('--n_actions', type=int, default=3)
    args = parser.parse_args()
    return args

# Define a wrapper to restrict the action space
class RestrictedActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env, n_actions):
        super().__init__(env)
        # Redefine action space to only include Left (0), Right (1), Forward (2)
        self.action_space = gym.spaces.Discrete(n_actions)  # 3 actions instead of 7
        # Mapping from new action indices to original MiniGrid actions
        self.action_mapping = {i:i for i in range(n_actions)}
        
    def step(self, action):
        # Map the restricted action to the original action
        mapped_action = self.action_mapping[action]
        # Call the original environment's step function
        obs, reward, terminated, truncated, info = self.env.step(mapped_action)
        return obs, reward, terminated, truncated, info

# -------------------------------
# Main training loop for offline batches
# -------------------------------
def main():
    args = parse_args()
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize wandb.
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
    )
    config = wandb.config

    global_steps = 0

    # Create the environment.
    # We use MiniGrid with image-only observations via ImgObsWrapper.
    base_env = gym.make(args.env_name)
    base_env = RestrictedActionSpaceWrapper(base_env, args.n_actions)  # restrict action space to Left, Right, Forward
    base_env = RecordEpisodeStatistics(base_env)
    base_env = ImgObsWrapper(base_env)
    # Optionally, check env compliance.
    check_env(base_env, warn=True)
    
    print('env is :', args.env_name)
    print('number of actions:', base_env.action_space.n)

    # Use CPU for offline training.
    device = "cpu"
    print(f"Using device: {device}")

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": config.features_dim},
        "optimizer_class": SGD,
    }
    
    # Initialize PPO with CnnPolicy.
    model = PPO(
        "CnnPolicy",
        base_env,
        verbose=1,
        device=device,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        batch_size=config.batch_size,
        tensorboard_log="./tb_logs/",
        seed=args.seed,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs=policy_kwargs
    )
    
    if args.initial_policy:
        import copy
        model_init = PPO.load(args.initial_policy, device=device, seed=args.seed)
        model.policy = copy.deepcopy(model_init.policy)
        print(f"Loaded initial policy from {args.initial_policy}")
    
    print("normalize_advantage:", model.normalize_advantage)
    model.policy.train()  # set training mode

    # Save initial checkpoint.
    model.save(os.path.join(args.save_path, f"policy_grad_{args.batch_begin_idx}.zip"))
    model._update_counter = args.batch_begin_idx

    # Offline training loop: load batches and perform manual gradient updates.
    for t in tqdm(range(args.batch_begin_idx, args.n_batches)):
        batch_path = osp.join(args.batches_dir, f"batch_{t+1}.pkl")
        with open(batch_path, "rb") as f:
            rollout_data = pickle.load(f)
            
        if rollout_data is None:
            print(f'encountering None batch at t={t}, exiting')
            # Save checkpoint every 320 batches.
            rollout_idx = args.n_batches // 320
            cp_path = osp.join(args.save_path, f"policy_after_rollout_{rollout_idx}.zip")
            model.save(cp_path)

            model.save(osp.join(args.save_path, f"policy_grad_{args.n_batches}.zip"))
            break
        
        # Schedule clip_range based on progress.
        progress = 1 - ((int(t/320) + 1) / (args.total_updates / 320))
        # If clip_range is a callable (schedule), use it; else, use constant value.
        clip_range = model.clip_range(progress) if callable(model.clip_range) else model.clip_range

        # Unpack rollout data.
        if isinstance(rollout_data, dict):
            actions = rollout_data["actions"]
            observations = rollout_data["observations"]
            advantages = rollout_data["advantages"]
            returns = rollout_data["returns"]
            old_log_prob = rollout_data["old_log_prob"]
        else:
            actions = rollout_data.actions
            observations = rollout_data.observations
            advantages = rollout_data.advantages
            returns = rollout_data.returns
            old_log_prob = rollout_data.old_log_prob
            
        actions = actions.to(device)
        observations = observations.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        old_log_prob = old_log_prob.to(device)

        # If action space is discrete, convert actions.
        if isinstance(model.action_space, spaces.Discrete):
            actions = actions.long().flatten()
        
        batch_size = len(observations)

        # Evaluate current policy on the batch.
        values, log_prob, entropy = model.policy.evaluate_actions(observations, actions)
        values = values.flatten()
        
        if model.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        ratio = th.exp(log_prob - old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        
        value_loss = F.mse_loss(returns, values)
        if entropy is None:
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)
        
        loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
        
        # Optimization step.
        model.policy.optimizer.zero_grad()
        loss.backward()
        # Scale gradients (if desired).
        for param in model.policy.parameters():
            if param.grad is not None:
                param.grad.data *= batch_size / config.batch_size
        model.policy.optimizer.step()
        
        # Save checkpoint every 320 batches.
        if (t + 1) % 320 == 0:
            rollout_idx = (t + 1) // 320
            cp_path = osp.join(args.save_path, f"policy_after_rollout_{rollout_idx}.zip")
            model.save(cp_path)
        model.save(osp.join(args.save_path, f"policy_grad_{t+1}.zip"))
    
    final_model_path = osp.join(args.save_path, "final_policy.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()

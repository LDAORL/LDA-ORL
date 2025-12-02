import argparse
import os
import os.path as osp
import pickle
import numpy as np
np.float_ = np.float64
import torch
import torch.nn.functional as F
import torch as th
from torch.optim import SGD, Adam
from tqdm import tqdm

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics


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
        description="Offline PPO training on LunarLander-v2 with wandb logging and batch checkpointing."
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
    
    # Environment hyperparameters.
    parser.add_argument('--env_name', type=str, default="LunarLander-v2",
                        help="Name of the environment (default: LunarLander-v2)")
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
    parser.add_argument("--project_name", type=str, default="lunarlander-rl",
                        help="WandB project name.")
    parser.add_argument("--run_name", type=str, default="ppo_lunarlander_offline_run",
                        help="WandB run name.")
    parser.add_argument('--optimizer_class', type=str, default="SGD")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_policy", type=str, default='',
                        help="Path to an initial policy checkpoint.")
    
    args = parser.parse_args()
    return args

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
    if args.env_name == 'FrozenLake-v1':
        from gymnasium.envs.toy_text.frozen_lake import generate_random_map
        env = gym.make(args.env_name, map_name='4x4', is_slippery=False)
    elif args.env_name == "highway-v0":
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
    elif "merge" in args.env_name:
        import highway_env
        env = gym.make(args.env_name, render_mode=None)
    else:
        env = gym.make(args.env_name)
    env = RecordEpisodeStatistics(env)
    # Optionally, check env compliance.
    # check_env(env, warn=True)
    
    print('env is :', args.env_name)
    print('observation space:', env.observation_space)
    # print('number of actions:', env.action_space.n)

    # Use CPU for offline training.
    device = "cpu"
    print(f"Using device: {device}")

    # For LunarLander, we use MlpPolicy so no custom feature extractor is needed.
    policy_kwargs = {
        "optimizer_class": SGD if args.optimizer_class == "SGD" else Adam,
    }
    
    # Initialize PPO with MlpPolicy.
    model = PPO(
        "MlpPolicy",
        env,
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
    print(model.policy.optimizer)  # Print the optimizer to verify it's set correctly
    
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
            print(f'encountered None batch at t={t}, exiting')
            rollout_idx = args.n_batches // 320
            cp_path = osp.join(args.save_path, f"policy_after_rollout_{rollout_idx}.zip")
            model.save(cp_path)
            model.save(osp.join(args.save_path, f"policy_grad_{args.n_batches}.zip"))
            break
        
        # Schedule clip_range based on progress.
        progress = 1 - ((int(t/320) + 1) / (args.total_updates / 320))
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
        # Scale gradients if desired.
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

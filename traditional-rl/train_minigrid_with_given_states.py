import argparse
import os
import numpy as np
import torch
from torch.optim import SGD
import wandb
import pickle  # For saving mini-batches
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback  # Requires wandb integration for SB3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import torch.nn as nn
import gymnasium as gym
import minigrid
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.wrappers import ImgObsWrapper

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

# =======================
# Environment creation
# =======================
def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        env = RecordEpisodeStatistics(env)  # for episode stats
        env = ImgObsWrapper(env)            # returns only the image observation (3,7,7)
        return env
    return _init

# =======================
# Callback to save states (observations) from rollouts
# =======================

class SaveStatesCallback(BaseCallback):
    """
    A callback that collects all states (observations) from the rollout buffer and
    saves them to disk when training ends.
    """
    def __init__(self, save_path, verbose=0):
        super(SaveStatesCallback, self).__init__(verbose)
        self.save_path = save_path
        self.all_states = []
    
    def _on_step(self) -> bool:
        # Not used in this callback, so simply return True.
        return True
    
    def on_rollout_end(self) -> None:
        # If the rollout buffer has observations, record them.
        if hasattr(self.model.rollout_buffer, "observations"):
            obs = self.model.rollout_buffer.observations.copy()
            self.all_states.extend(obs)
    
    def on_training_end(self) -> None:
        # Save the collected states to disk as a NumPy file.
        all_states_array = np.array(self.all_states)
        save_file = os.path.join(self.save_path, "all_states.npy")
        np.save(save_file, all_states_array)
        # print(f"Saved all states to {save_file}")

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
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per rollout.")
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
    
    parser.add_argument('--initial_policy', type=str, default='', required=True)
    parser.add_argument('--rollout_begin_idx', type=int, default=0)

    parser.add_argument("--total_rollouts", type=int, default=20,
                        help="Number of fixed rollouts provided.")
    parser.add_argument('--env_name', type=str, default="MiniGrid-Empty-5x5-v0")
    return parser.parse_args()

# =======================
# Main training function
# =======================

def main():
    # Parse hyperparameters from command line.
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)  # Create the save directory if it doesn't exist.

    # Initialize wandb for experiment tracking.
    # Note: We enable TensorBoard syncing (even if you don't manually use TensorBoard)
    # so that SB3 logs are picked up and sent to WandB.
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args),
        sync_tensorboard=True,  # This enables syncing SB3's TensorBoard logs to WandB.
        save_code=True,         # Optionally, save code to wandb.
    )
    config = wandb.config

    # Create environment wrapped in DummyVecEnv and VecMonitor
    env = DummyVecEnv([make_env(args.env_name)])
    env = VecMonitor(env)
    
    # Define policy keyword arguments
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config.features_dim),
        optimizer_class=th.optim.SGD if args.optimizer_class == "SGD" else th.optim.Adam,
    )

    # Compute total_global_steps as the sum of lengths of all fixed rollouts.
    total_timesteps = args.total_timesteps
    global_steps = args.n_steps * args.rollout_begin_idx  # Initialize global steps counter
    
    # Use GPU if available.
    # device = "cuda:0"
    device = "cpu"
    print(f"Using device: {device}")

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=config.learning_rate,
        tensorboard_log="./tb_logs/",
        seed=args.seed,
        policy_kwargs=policy_kwargs
    )
    
    if args.initial_policy:
        import copy
        model_init = PPO.load(
            args.initial_policy, 
            device=device, 
            # seed=args.seed, 
            # policy_kwargs=policy_kwargs
        )
        model.policy = copy.deepcopy(model_init.policy)
        print('loaded model with initial policy done', args.initial_policy)


    # Initialize a counter for gradient updates; this will be used to index both batches and checkpoints.
    model._update_counter = 320 * args.rollout_begin_idx
    model.save(os.path.join(args.save_path, f"policy_grad_{model._update_counter}.zip"))

    model.num_timesteps = global_steps

    # Patch the optimizer's step method to save the checkpoint (pi_t) after each gradient update.
    original_optimizer_step = model.policy.optimizer.step
    def patched_optimizer_step(*args, **kwargs):
        result = original_optimizer_step(*args, **kwargs)
        # Use the same counter as set in the patched get() method.
        checkpoint_filename = os.path.join(save_path, f"policy_grad_{model._update_counter}.zip")
        model.save(checkpoint_filename)
        # print(f"Saved checkpoint: {checkpoint_filename}")
        return result
    model.policy.optimizer.step = patched_optimizer_step

    # Set up a global clip_range function.
    # Retrieve the original clip_range schedule function from the model.
    # Typically, model.clip_range is a function that accepts a progress value.
    # def global_clip_range(dummy_progress):
    #     # Compute global progress as 1 - (global_steps / total_timesteps)
    #     global_progress = 1.0 - (global_steps / total_timesteps)
    #     return config.clip_range * global_progress  # or use original_schedule if available

    # # Override the model's clip_range with our global version.
    # model.clip_range = global_clip_range

    # Create your state-saving callback.
    save_states_callback = SaveStatesCallback(save_path=save_path, verbose=1)
    callback = CallbackList([WandbCallback(verbose=2), save_states_callback])
    
    # Now loop over each fixed rollout.
    total_rollouts = args.total_rollouts
    
    print('===============================================================')
    print(f'begin training at rollout_idx = {args.rollout_begin_idx} and global_steps = {global_steps}')
    print(f'target rollouts = {total_rollouts} with the global_steps = {total_timesteps}')
    
    
    for rollout_idx in range(args.rollout_begin_idx, total_rollouts):
        print(f"Training on rollout {rollout_idx+1}/{total_rollouts} with {model.n_steps} transitions.")
        
        # Patch the rollout buffer's get() method to save each mini-batch (B_t).
        original_get = model.rollout_buffer.get
        def patched_get(batch_size):
            for mini_batch in original_get(batch_size):
                model._update_counter += 1
                batch_filename = os.path.join(save_path, f"batch_{model._update_counter}.pkl")
                with open(batch_filename, "wb") as f:
                    pickle.dump(mini_batch, f)
                # print(f"Saved batch: {batch_filename}")
                yield mini_batch
        model.rollout_buffer.get = patched_get

        # Update global steps.
        global_steps += model.n_steps
        print(f"Global progress: {1.0 - (global_steps / total_timesteps):.3f}")

        # Call learn() with total_timesteps equal to the length of the fixed rollout.
        model.learn(
            total_timesteps=args.total_timesteps,
            upper_timesteps=model.num_timesteps+args.n_steps,
            callback=callback,
            reset_num_timesteps=2,
        )        
        
        # Optionally, save a checkpoint after each rollout.
        model.save(os.path.join(save_path, f"policy_after_rollout_{rollout_idx+1}.zip"))

    final_model_path = os.path.join(save_path, "final_policy.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()

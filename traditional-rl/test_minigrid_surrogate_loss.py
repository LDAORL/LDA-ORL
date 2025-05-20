import argparse
from stable_baselines3 import PPO
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='easy_to_hard_40k_ckpt_batch')
parser.add_argument('--rollout_file', type=str, required=True)
args = parser.parse_args()

def compute_surrogate_validation_loss(model, sample):
    """
    Compute the surrogate validation loss using a fixed action target.
    
    The sample is a dictionary that contains:
      - 'observations': observations (e.g. image data)
      - 'actions': target actions (as integers) used to compute the advantage
      - 'advantages': advantage estimates computed by the oracle model
    
    This function computes the loss:
        L = -mean( advantage * log(pi(current_model)(target_action | observation)) )
    and backpropagates to compute gradients on the current model's policy.
    
    Returns:
      grad_dict: a dictionary mapping parameter names to their gradient tensors.
      loss_val: the computed scalar loss value.
    """
    device = model.device

    # Convert the sample components to torch tensors on the model device.
    obs = torch.tensor(sample["observations"], dtype=torch.float32, device=device)
    target_actions = torch.tensor(sample["actions"], dtype=torch.long, device=device)
    advantages = torch.tensor(sample["advantages"], dtype=torch.float32, device=device)
    
    # Ensure observations have a batch dimension.
    if obs.ndim == len(model.policy.observation_space.shape):
        obs = obs.unsqueeze(0)
    
    # Ensure target_actions is 1D (batch size)
    if target_actions.ndim == 0:
        target_actions = target_actions.unsqueeze(0)
    elif target_actions.ndim > 1:
        target_actions = target_actions.flatten()

    # Zero gradients.
    model.policy.optimizer.zero_grad()
    
    # Get the current model's distribution from the observations.
    dist = model.policy.get_distribution(obs)
    logits = dist.distribution.logits  # shape: [batch, num_actions]
    
    # Compute log probabilities.
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log probability corresponding to the target action.
    selected_log_probs = log_probs.gather(1, target_actions.unsqueeze(1)).squeeze(1)
    
    # Compute the surrogate loss: negative mean advantage-weighted log probability.
    # loss = -torch.mean(advantages * selected_log_probs)
    loss = -torch.mean(advantages.squeeze() * selected_log_probs)
    
    return loss.item()


# Load the rollout buffer.
rollouts = pickle.load(open(args.rollout_file, "rb"))

# Extract a sample from the rollout buffer.
# (Assumes the rollout buffer has fields: observations, actions, advantages)
sample = {
    "observations": rollouts.observations.squeeze(1),
    "actions": rollouts.actions,
    "advantages": rollouts.advantages,
}

print('rollout sample size =', sample['observations'].shape[0])

model = PPO.load(args.model_path)
print('loaded model from', args.model_path)
loss = compute_surrogate_validation_loss(model, sample)
print('Surrogate validation loss:', loss)
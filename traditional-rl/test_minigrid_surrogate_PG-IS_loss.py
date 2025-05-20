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
parser.add_argument('--ref_policy', type=str, required=True)
parser.add_argument('--rollout_file', type=str, required=True)
args = parser.parse_args()

def get_action_prob(model, obs, actions):
    """
    Get the action probabilities for a given set of observations and actions.
    
    Parameters:
      model: a PPO model (from stable-baselines3) whose policy is an ActorCriticPolicy.
      obs: observations (torch tensor).
      actions: actions (torch tensor).
      
    Returns:
      selected_probs: the log probabilities of the selected actions.
    """
    # # Get the distribution from the model's policy.
    # dist = model.policy.get_distribution(obs)
    
    # # Compute log probabilities for the selected actions.
    # log_probs = dist.distribution.log_prob(actions)
    
    # Get the current model's distribution from the observations.
    print('obs', obs.shape)
    dist = model.policy.get_distribution(obs)
    logits = dist.distribution.logits  # shape: [batch, num_actions]

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    selected_probs = torch.exp(selected_log_probs)
    return selected_log_probs, selected_probs

def compute_surrogate_validation_loss(model, sample, model_ref):
    """
    Compute a REINFORCE-style surrogate validation loss.
    
    For each episode, we compute the total return:
        R_ep = sum_{t in episode} reward[t]
    and the average log–probability:
        avg_logp_ep = (1/T_ep) * sum_{t in episode} log(pi(a_t | s_t))
    
    Then the per-episode loss is defined as:
        L_ep = - R_ep * avg_logp_ep
    and the final loss is the mean over episodes.
    
    The sample dictionary must contain:
      - 'observations': observations (e.g., image data)
      - 'actions': target actions (as integers)
      - 'episode_starts': boolean array indicating the start of a new episode
      
    Returns:
      grad_dict: a dictionary mapping policy parameter names to their gradient tensors.
      loss_val: the computed scalar loss value.
    """
    device = model.device

    # Convert sample components to torch tensors.
    obs = torch.tensor(sample["observations"], dtype=torch.float32, device=device)
    target_actions = torch.tensor(sample["actions"], dtype=torch.long, device=device)
    advantages = torch.tensor(sample["advantages"], dtype=torch.float32, device=device)
    
    # print('before transformation', obs.shape)
    # # Flatten observations if they are grouped (e.g., shape [num_episodes, T, ...]).
    # expected_ndim = len(model.policy.observation_space.shape) + 1  # [N, ...]
    # if obs.ndim > expected_ndim:
    #     obs = obs.view(-1, *obs.shape[-len(model.policy.observation_space.shape):])
    
    # if obs.ndim == len(model.policy.observation_space.shape):
    #     obs = obs.unsqueeze(0)
    
    print('before transformation', obs.shape)
    obs_shape = model.policy.observation_space.shape  # could be () or (4,) etc.
    expected_ndim = len(obs_shape) + 1  # e.g., 2 for vector obs, 1 for scalar obs

    # Special Case: if obs_shape == () and obs is scalar (e.g., FrozenLake), flatten properly
    if obs_shape == () and obs.shape[1] == 1:
        obs = obs.squeeze(1)
        # pass
        
    else:
        # Case 1: too many dims → flatten grouped episode batches
        if obs.ndim > expected_ndim:
            obs = obs.view(-1, *obs.shape[-len(obs_shape):])

        # Case 2: too few dims → unsqueeze leading batch dim
        elif obs.ndim == len(obs_shape):
            obs = obs.unsqueeze(0)
        
    if target_actions.ndim > 1:
        target_actions = target_actions.flatten()
    
    if advantages.ndim > 1:
        advantages = advantages.flatten()
    

    with torch.no_grad():
        selected_log_probs, selected_probs = get_action_prob(model, obs, target_actions)
        _, selected_probs_ref = get_action_prob(model_ref, obs, target_actions)

    print(advantages.shape, selected_log_probs.shape, selected_probs.shape, selected_probs_ref.shape)
        
    loss = -torch.mean(advantages * selected_log_probs * selected_probs / selected_probs_ref)

    return loss.item()


# Load the rollout buffer.
rollouts = pickle.load(open(args.rollout_file, "rb"))

# Extract a sample from the rollout buffer.
# (Assumes the rollout buffer has fields: observations, actions, advantages)
if type(rollouts) == dict:
    sample = rollouts
    sample['observations'] = sample['observations'].squeeze(1)
else:
    sample = {
        "observations": rollouts.observations.squeeze(1),
        "actions": rollouts.actions,
        "advantages": rollouts.advantages,
    }

print('rollout sample size =', sample['observations'].shape[0])

model = PPO.load(args.model_path, device="cpu")
model_ref = PPO.load(args.ref_policy, device="cpu")

model.policy.set_training_mode(False)
model_ref.policy.set_training_mode(False)

print('loaded model from', args.model_path)
print('loaded reference model from', args.ref_policy)

loss = compute_surrogate_validation_loss(model, sample, model_ref)
print('Surrogate validation loss:', loss)
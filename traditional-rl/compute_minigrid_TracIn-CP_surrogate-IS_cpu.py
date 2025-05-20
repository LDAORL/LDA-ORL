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
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_updates', type=int, default=6400)
parser.add_argument('--total_updates', type=int, default=6400)
parser.add_argument('--begin_step', type=int, default=0)
parser.add_argument('--rollout_file', type=str, required=True)
parser.add_argument('--ref_policy', type=str, required=True)
parser.add_argument('--env_short', type=str, default='minigrid')
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
    dist = model.policy.get_distribution(obs)
    logits = dist.distribution.logits  # shape: [batch, num_actions]

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    selected_probs = torch.exp(selected_log_probs)
    return selected_log_probs, selected_probs

# valid_loss = []
def compute_surrogate_validation_loss(model, obs, target_actions, advantages, selected_probs_ref):
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
    # Zero gradients.
    model.policy.optimizer.zero_grad()
    
    with torch.no_grad():
        _, selected_probs_no_grad = get_action_prob(model, obs, target_actions)
        
    selected_log_probs, _ = get_action_prob(model, obs, target_actions)
    
    # print(advantages.shape, selected_log_probs.shape, selected_probs_no_grad.shape, selected_probs_ref.shape)
    
    # Compute the surrogate loss: negative mean advantage-weighted log probability.
    # loss = -torch.mean(advantages * selected_log_probs)
    loss = -torch.mean(advantages * selected_log_probs * selected_probs_no_grad / selected_probs_ref)

    # Backward pass.
    loss.backward()
    
    # valid_loss.append(loss.item())
    
    # Collect gradients for each policy parameter.
    grad_dict = {}
    for name, param in model.policy.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_dict[name] = param.grad.detach().clone()
    
    return grad_dict


# train_loss = []
# all_adjusted_advantages = []
def compute_ppo_policy_gradient(model, sample, progress, normalize_advantages=True, index=-1):
    """
    Compute the gradient of the PPO surrogate loss on a given sample using the policy network.
    
    This function uses the clipped surrogate objective plus an entropy bonus, similar to how PPO
    computes its loss. It assumes that `sample` is a dictionary (or object with attributes)
    that includes:
      - 'observations': the observation(s) (shape: [N, ...] or [N] for a single sample)
      - 'actions': the corresponding actions
      - 'old_log_prob': the log probability computed when the sample was collected
      - 'advantages': the advantage values for the sample(s)
      
    Parameters:
      model: a PPO model (from stable-baselines3) whose policy is an ActorCriticPolicy.
      sample: a dict-like object containing at least the keys mentioned above.
      
    Returns:
      grad_dict: a dictionary mapping parameter names (for the policy network only)
                 to their computed gradient tensors.
    """
    # Convert the sample to tensors and ensure a batch dimension.
    if isinstance(sample, dict):
        obs = sample['observations']
        act = sample['actions']
        old_log_prob = sample['old_log_prob']
        advantages = sample['advantages']
    else:
        obs = sample.observations
        act = sample.actions
        old_log_prob = sample.old_log_prob
        advantages = sample.advantages
        
    obs = obs.to(model.device)
    act = act.to(model.device)
    old_log_prob = old_log_prob.to(model.device)
    advantages = advantages.to(model.device)
    
    # If the observation is not batched, add a batch dimension.
    if len(obs.shape) == len(model.policy.observation_space.shape):
        obs = obs[None, ...]
    # if isinstance(act, int):
    #     act = torch.tensor([act], dtype=torch.int64, device=model.device)
    # else:
    act = torch.tensor(act, dtype=torch.int64, device=model.device)
    
    # Zero gradients.
    model.policy.optimizer.zero_grad()
    
    # Evaluate current log probabilities (and entropy) for the given observation-action pair.
    # Note: evaluate_actions expects the inputs to have a batch dimension.
    values, new_log_prob, entropy = model.policy.evaluate_actions(obs, act.long().flatten())
    # values = values.flatten()
    
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Compute the probability ratio.
    ratio = torch.exp(new_log_prob - old_log_prob)
    
    clip_range = model.clip_range(progress)
    
    # Compute the surrogate objective (clipped).
    policy_loss_1 = ratio * advantages
    policy_loss_2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    
    # assert torch.equal(new_log_prob, old_log_prob)
    # assert torch.equal(policy_loss_1, policy_loss_2)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
    # all_adjusted_advantages.append(advantages.mean().item())
    
    # value_loss = F.mse_loss(returns, values)
    
    # Include an entropy bonus (using the model's ent_coef parameter).
    entropy_loss = -entropy.mean()
    
    # Total loss (only policy part; we ignore value loss here).
    # loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
    loss = policy_loss + model.ent_coef * entropy_loss
    
    # Backward pass to compute gradients.
    loss.backward()
    
    # train_loss.append(loss.item())
    
    # Collect gradients for parameters in the policy network (exclude value network parameters).
    grad_dict = {}
    for name, param in model.policy.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Exclude any parameters that are part of the value network.
            if ("value_net" not in name) and ("mlp_extractor.value_net" not in name):
                grad_dict[name] = param.grad.detach().clone()
    
    return grad_dict

# Example of extracting per-sample gradients from a RolloutBufferSamples object:
def compute_gradients_for_train_batch(model, train_batch, progress):
    """
    Given a RolloutBufferSamples object (train_batch), compute per-sample gradients.

    Returns:
      grad_list_train: a list where each element is the gradient dictionary for one sample.
    """
    grad_list_train = []
    # Assume train_batch.observations and train_batch.actions are arrays with the first dimension equal to the batch size.
    batch_size = train_batch.observations.shape[0]
    adjusted_advantages = (train_batch.advantages - train_batch.advantages.mean()) / (train_batch.advantages.std() + 1e-8)
    for i in range(batch_size):
        # Extract the i-th sample.
        sample = {
            'observations': train_batch.observations[i].unsqueeze(0),
            'actions': train_batch.actions[i].unsqueeze(0),
            'old_log_prob': train_batch.old_log_prob[i].unsqueeze(0),
            'advantages': adjusted_advantages[i].unsqueeze(0),
            # 'returns': train_batch.returns[i].unsqueeze(0)
        }
        grad = compute_ppo_policy_gradient(model, sample, progress, normalize_advantages=False, index=i)
        grad_list_train.append(grad)
        
    # pickle.dump(grad_list_train, open('grad_list_train.pkl', 'wb'))
    return grad_list_train

def flatten_gradients(grad_dict):
    """
    Flatten a dictionary of gradients into a single 1D tensor.
    """
    return torch.cat([g.view(-1) for g in grad_dict.values()])

def inner_product_gradients(grad1, grad2):
    """
    Compute the inner product between two gradient dictionaries.
    """
    flat1 = flatten_gradients(grad1)
    flat2 = flatten_gradients(grad2)
    return torch.dot(flat1, flat2)


def flatten_ob_act_tensors(ob, act):
    return tuple(np.concatenate((ob.cpu().numpy().flatten(), act.cpu().numpy().flatten())))

def flatten_ob_act_prob_adv(ob, act, prob, adv):
    return tuple(np.concatenate((ob.cpu().numpy().flatten(), act.cpu().numpy().flatten(), prob.cpu().numpy().flatten(), adv.cpu().numpy().flatten())))

def flatten_ob_act_prob_adv_rew_ret(ob, act, prob, adv, rew, ret):
    return tuple(np.concatenate((ob.cpu().numpy().flatten(), act.cpu().numpy().flatten(), prob.cpu().numpy().flatten(), adv.cpu().numpy().flatten(), rew.cpu().numpy().flatten(), ret.cpu().numpy().flatten())))

def update_scores_for_samples(batch, scores):
    global d_global
    batch_size = batch.observations.shape[0]
    for i in range(batch_size):
        # tpl = flatten_ob_act_tensors(batch.observations[i], batch.actions[i])
        tpl = flatten_ob_act_prob_adv(batch.observations[i], batch.actions[i], batch.old_log_prob[i], batch.advantages[i])
        # print(tpl)
        if tpl not in d_global:
            d_global[tpl] = []

        d_global[tpl].append(scores[i])
        
requires_grad_params = set([
    'mlp_extractor.policy_net.0.weight',
    'mlp_extractor.policy_net.0.bias',
    'mlp_extractor.policy_net.2.weight',
    'mlp_extractor.policy_net.2.bias',
    'action_net.weight',
    'action_net.bias'
])

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

if args.ref_policy.endswith('.zip'):
    # Load the reference policy from a zip file.
    model_ref = PPO.load(args.ref_policy, device="cpu")
else:
    model_ref = PPO.load(f'./models_{args.env_short}/{args.model_path}/policy_grad_{args.ref_policy}.zip', device="cpu")

d_global = {}
all_scores = []

device = model_ref.device
# Convert sample components to torch tensors.
obs = torch.tensor(sample["observations"], dtype=torch.float32, device=device)
target_actions = torch.tensor(sample["actions"], dtype=torch.long, device=device)
advantages = torch.tensor(sample["advantages"], dtype=torch.float32, device=device)

# # Flatten observations if they are grouped (e.g., shape [num_episodes, T, ...]).
# expected_ndim = len(model_ref.policy.observation_space.shape) + 1  # [N, ...]
# if obs.ndim > expected_ndim:
#     obs = obs.view(-1, *obs.shape[-len(model_ref.policy.observation_space.shape):])

# if obs.ndim == len(model_ref.policy.observation_space.shape):
#     obs = obs.unsqueeze(0)

obs_shape = model_ref.policy.observation_space.shape  # could be () or (4,) etc.
expected_ndim = len(obs_shape) + 1  # e.g., 2 for vector obs, 1 for scalar obs

# Special Case: if obs_shape == () and obs is scalar (e.g., FrozenLake), flatten properly
if obs_shape == () and obs.shape[1] == 1:
    obs = obs.squeeze(1)
    
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
    _, selected_probs_ref = get_action_prob(model_ref, obs, target_actions)

grad_ref = compute_surrogate_validation_loss(model_ref, obs, target_actions, advantages, selected_probs_ref)

# print('grad ref norm=', torch.norm(flatten_gradients(grad_ref)).item())

for i in tqdm(range(args.begin_step, args.begin_step + 32)):
    batch_i = pickle.load(open(f'./models_{args.env_short}/{args.model_path}/batch_{i+1}.pkl', 'rb'))
    
    # if i == 0:
    #     obs_0 = batch_i.observations[0]
    #     act_0 = batch_i.actions[0]
    
    progress = 1-(int(i/320)+1)/(args.total_updates/320)

    grad_i = compute_gradients_for_train_batch(model_ref, batch_i, progress)
    scores = [inner_product_gradients(grad, grad_ref).item() for grad in grad_i]
    # pickle.dump(grad_full, open(f'grad_full.pkl', 'wb'))

    all_scores += scores
    
    update_scores_for_samples(batch_i, scores)
    
os.makedirs(args.output_dir, exist_ok=True)
pickle.dump({
    'all_scores': all_scores,
    'd_global': d_global,
    # 'train_loss': train_loss,
    # 'valid_loss': valid_loss,
    # 'obs_0': obs_0,
    # 'act_0': act_0,
    # 'all_adjusted_advantages': all_adjusted_advantages,
}, open(os.path.join(args.output_dir, 'scores.pkl'), 'wb'))

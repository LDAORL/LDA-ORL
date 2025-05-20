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
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
    
    # value_loss = F.mse_loss(returns, values)
    
    # Include an entropy bonus (using the model's ent_coef parameter).
    entropy_loss = -entropy.mean()
    
    # Total loss (only policy part; we ignore value loss here).
    # loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
    loss = policy_loss + model.ent_coef * entropy_loss
    
    # Backward pass to compute gradients.
    loss.backward()
    

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
    
    
def add_hooks_to_model(model, _xs, _gs):
    # register hooks for the model
    for name, module in model.policy.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'value_net' in name:
                continue

            # we'll store val‐hooks and train‐activations here
            _xs[name] = None
            _gs[name] = None

            # forward hook: save x (pre‐activation) on forward
            def fwd(m, inp, out, nm=name):
                # print('fwd', nm)
                x, = inp
                _xs[nm] = x.detach()            # [B, …, in_features]
            module.register_forward_hook(fwd)

            # backward hook: save grad w.r.t. output on backward
            def bwd(m, grad_in, grad_out, nm=name):
                # print('bwd', nm)
                g, = grad_out
                _gs[nm] = g.detach()           # [B, …, out_features]
            module.register_full_backward_hook(bwd)
    

def compute_sample_ip_layers_vec(train_xs, train_gs, val_xs, val_gs):
    layers = list(train_xs.keys())
    n_train = next(iter(train_xs.values())).shape[0]
    sample_IP = torch.zeros(n_train, dtype=next(iter(train_xs.values())).dtype)
    for layer in layers:
        X_t = train_xs[layer]  # [n_train, F]
        G_t = train_gs[layer]  # [n_train, G]
        X_v = val_xs[layer]    # [n_val, F]
        G_v = val_gs[layer]    # [n_val, G]
        # sum over val j: P = sum_j g_v[j].T @ x_v[j] -> [G, F]
        P = G_v.transpose(0,1) @ X_v  # [G, F]
        # Q_i = g_t[i].T @ x_t[i] -> [G, F] for each i. Vectorize:
        Q = G_t.unsqueeze(2) * X_t.unsqueeze(1)  # [n_train, G, F]
        # inner products:
        sample_IP += (Q * P.unsqueeze(0)).sum(dim=(1,2))  # [n_train]
        
        bias_sum = G_v.sum(dim=0)
        sample_IP += (G_t * bias_sum.unsqueeze(0)).sum(dim=1)
    return sample_IP


for i in tqdm(range(args.begin_step, args.num_updates)):
    model_i = PPO.load(f'./models_{args.env_short}/{args.model_path}/policy_grad_{i}.zip', device="cpu")
    
    _xs, _gs = {}, {}
    add_hooks_to_model(model_i, _xs, _gs)
    
    batch_i = pickle.load(open(f'./models_{args.env_short}/{args.model_path}/batch_{i+1}.pkl', 'rb'))
    
    progress = 1-(int(i/320)+1)/(args.total_updates/320)

    compute_surrogate_validation_loss(model_i, obs, target_actions, advantages, selected_probs_ref)
    _val_xs, _val_gs = _xs, _gs
    _xs, _gs = {}, {}
    
    add_hooks_to_model(model_i, _xs, _gs)
    
    compute_ppo_policy_gradient(model_i, batch_i, progress, normalize_advantages=True)
    _train_xs, _train_gs = _xs, _gs
    
    out = compute_sample_ip_layers_vec(_train_xs, _train_gs, _val_xs, _val_gs)
    
    scores = out.cpu().tolist()
    
    all_scores += scores
    
    epoch_id = i // 32
    update_scores_for_samples(batch_i, scores)
    _xs, _gs = {}, {}
    
os.makedirs(args.output_dir, exist_ok=True)
pickle.dump({
    'all_scores': all_scores,
    'd_global': d_global
}, open(os.path.join(args.output_dir, 'scores.pkl'), 'wb'))

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
parser.add_argument('--sanity_check', action='store_true', default=False)
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

def load_in_train_data():
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

    return obs, target_actions, advantages

    

# load in model and add hooks
if args.ref_policy.endswith('.zip'):
    # Load the reference policy from a zip file.
    model_ref = PPO.load(args.ref_policy, device="cpu")
else:
    model_ref = PPO.load(f'./models_{args.env_short}/{args.model_path}/policy_grad_{args.ref_policy}.zip', device="cpu")

if args.sanity_check:    
    print(model_ref.policy)

# load in data
obs, act, advantages = load_in_train_data()

# register hooks for the model
_xs, _gs = {}, {}
conv_params = {}
for name, module in model_ref.policy.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        if 'value_net' in name:
            continue
        print('registering hook for', name)
        
        if isinstance(module, torch.nn.Conv2d):
            conv_params[name] = {
                'kernel_size': module.kernel_size,
                'stride':      module.stride,
                'padding':     module.padding
            }        

        # we'll store val‐hooks and train‐activations here
        _xs[name] = None
        _gs[name] = None

        # forward hook: save x (pre‐activation) on forward
        def fwd(m, inp, out, nm=name):
            x, = inp
            # for Conv2d this is [B,Cin,H,W], for Linear [B,Fin]
            _xs[nm] = x.detach()
        module.register_forward_hook(fwd)

        # backward hook: save grad w.r.t. output on backward
        def bwd(m, grad_in, grad_out, nm=name):
            g, = grad_out
            # for Conv2d this is [B,Cout,H',W'], for Linear [B,Fout]
            _gs[nm] = g.detach()
        module.register_full_backward_hook(bwd)

############################
##### forward & backward on training
############################
# adjusted_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Break the sequence into chunks of length 64 and normalize within each chunk using vectorized operations
chunk_size = 64
num_chunks = 32

# Reshape into chunks and compute mean and std along the chunk dimension
chunks = advantages.view(num_chunks, chunk_size)
chunk_means = chunks.mean(dim=1, keepdim=True)
chunk_stds = chunks.std(dim=1, keepdim=True) + 1e-8

# Normalize each chunk
normalized_chunks = (chunks - chunk_means) / chunk_stds

# Flatten back and remove the padding
adjusted_advantages = normalized_chunks.view(-1)[:advantages.shape[0]]

# Zero gradients.
model_ref.policy.optimizer.zero_grad()

# Evaluate current log probabilities (and entropy) for the given observation-action pair.
# Note: evaluate_actions expects the inputs to have a batch dimension.
values, new_log_prob, entropy = model_ref.policy.evaluate_actions(obs, act.long().flatten())
# values = values.flatten()

# Compute the probability ratio.
ratio = torch.exp(new_log_prob - new_log_prob.detach())

# Compute the surrogate objective (clipped).
policy_loss_1 = ratio * adjusted_advantages
policy_loss = -policy_loss_1.sum()

# value_loss = F.mse_loss(returns, values)

# Include an entropy bonus (using the model's ent_coef parameter).
entropy_loss = -entropy.mean()

# Total loss (only policy part; we ignore value loss here).
# loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
loss = policy_loss + model_ref.ent_coef * entropy_loss

loss.backward()

train_loss = loss.item()

# Examine the status of the saved hooks
if args.sanity_check:
    for name in _xs:
        print(f"Layer: {name}")
        print(f"  Forward hook (activations): {_xs[name].shape if _xs[name] is not None else 'None'}")
        print(f"  Backward hook (gradients): {_gs[name].shape if _gs[name] is not None else 'None'}")


_train_xs, _train_gs = _xs, _gs
_xs, _gs = {}, {}

############################
##### forward & backward on validation
############################

model_ref.policy.optimizer.zero_grad()

selected_log_probs, _ = get_action_prob(model_ref, obs, act)
loss = -torch.mean(advantages * selected_log_probs)

loss.backward()

valid_loss = loss.item()

# Examine the status of the saved hooks
if args.sanity_check:
    for name in _xs:
        print(f"Layer: {name}")
        print(f"  Forward hook (activations): {_xs[name].shape if _xs[name] is not None else 'None'}")
        print(f"  Backward hook (gradients): {_gs[name].shape if _gs[name] is not None else 'None'}")

_val_xs, _val_gs = _xs, _gs

############################
##### compute the inner product
############################

def compute_sample_ip_layers_vec(train_xs, train_gs, val_xs, val_gs):
    """
    Compute influence inner-product for both Linear and Conv2d layers.

    train_xs, train_gs: dict layer_name → Tensor of train activations and grads
    val_xs,   val_gs:   dict layer_name → Tensor of val   activations and grads
    conv_params: global dict layer_name → {'kernel_size','stride','padding'}
    """
    layers = list(train_xs.keys())
    # number of train samples
    n_train = next(iter(train_xs.values())).shape[0]
    # initialize output
    sample_IP = torch.zeros(
        n_train,
        dtype=next(iter(train_xs.values())).dtype,
        device=next(iter(train_xs.values())).device
    )

    for layer in layers:
        X_t = train_xs[layer]
        G_t = train_gs[layer]
        X_v = val_xs[layer]
        G_v = val_gs[layer]

        # Linear layer case
        if X_t.dim() == 2:
            # weight-block
            P = G_v.transpose(0,1) @ X_v              # [G, F]
            Q = G_t.unsqueeze(2) * X_t.unsqueeze(1)   # [n_train, G, F]
            sample_IP += (Q * P.unsqueeze(0)).sum(dim=(1,2))
            # bias-block
            bias_sum = G_v.sum(dim=0)                 # [G]
            sample_IP += (G_t * bias_sum.unsqueeze(0)).sum(dim=1)

        # Conv2d layer case
        else:
            # get stored conv params
            p = conv_params[layer]
            # unfold val inputs → [n_val, Cin*kH*kW, L]
            patches_v = F.unfold(
                X_v,
                kernel_size=p['kernel_size'],
                stride=p['stride'],
                padding=p['padding']
            )
            n_val, C_out, H1, W1 = G_v.shape
            # flatten grads → [n_val, C_out, L]
            Gv_flat = G_v.view(n_val, C_out, -1)
            # weight-grads per val → [n_val, C_out, Cin*kH*kW]
            wg_v = torch.matmul(Gv_flat, patches_v.transpose(1,2))
            # aggregate val contributions
            P = wg_v.sum(dim=0)                       # [C_out, Cin*kH*kW]

            # train side unfold
            patches_t = F.unfold(
                X_t,
                kernel_size=p['kernel_size'],
                stride=p['stride'],
                padding=p['padding']
            )
            n_tr = G_t.shape[0]
            Gt_flat = G_t.view(n_tr, C_out, -1)
            wg_t = torch.matmul(Gt_flat, patches_t.transpose(1,2))  # [n_tr, C_out, Cin*kH*kW]
            sample_IP += (wg_t * P.unsqueeze(0)).sum(dim=(1,2))

            # bias term
            bias_sum = G_v.sum(dim=(0,2,3))           # [C_out]
            tr_bias = G_t.sum(dim=(2,3))              # [n_tr, C_out]
            sample_IP += (tr_bias * bias_sum.unsqueeze(0)).sum(dim=1)

    return sample_IP

out = compute_sample_ip_layers_vec(_train_xs, _train_gs, _val_xs, _val_gs)

os.makedirs(args.output_dir, exist_ok=True)
torch.save({
    # '_train_xs': _train_xs,
    # '_train_gs': _train_gs,
    # '_val_xs': _val_xs,
    # '_val_gs': _val_gs,
    # 'train_loss': train_loss,
    # 'valid_loss': valid_loss,
    # 'policy_loss': policy_loss_1.detach(),
    # 'adjusted_advantages': adjusted_advantages,
    # 'obs_0': obs[0],
    # 'act_0': act[0],
    'ip': out
}, os.path.join(args.output_dir, 'inner_product.pth'))



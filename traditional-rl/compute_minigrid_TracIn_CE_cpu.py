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
parser.add_argument('--valid_path', type=str, default='reference_minigrid_1_1_left.pkl')
parser.add_argument('--model_path', type=str, default='easy_to_hard_40k_ckpt_batch')
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_updates', type=int, default=6400)
parser.add_argument('--total_updates', type=int, default=6400)
parser.add_argument('--begin_step', type=int, default=0)
parser.add_argument('--env_short', type=str, default='minigrid')
args = parser.parse_args()

def compute_ce_policy_gradient(model, sample):
    """
    Compute the gradient of the cross entropy loss on a given sample using the policy network,
    treating the task as a classification problem.

    This function assumes that `sample` is a dictionary (or object with attributes) that includes:
      - 'observations': the observation(s) (shape: [N, ...] or a single observation)
      - 'actions': the groundtruth action(s) as integer(s)

    It performs the full forward pass using the policy's get_distribution method to obtain the 
    logits, computes the cross entropy loss, backpropagates, and returns a dictionary mapping 
    parameter names (for the policy network only) to their gradient tensors.
    """
    # Extract observation and groundtruth action from the sample.
    obs = sample['observations']
    gt_act = sample['actions']
        
    # Convert observation to a tensor and ensure it has a batch dimension.
    obs = obs.to(model.device)
    if len(obs.shape) == len(model.policy.observation_space.shape):
        obs = obs[None, ...]  # add batch dimension

    # Convert groundtruth action(s) to tensor.
    gt_act = torch.tensor(gt_act, dtype=torch.long, device=model.device)
    if len(gt_act.shape) == 0:
        gt_act = gt_act.unsqueeze(0)
    elif len(gt_act.shape) > 1:
        gt_act = gt_act.flatten()

    # Zero gradients.
    model.policy.optimizer.zero_grad()

    # Use the policy's built-in method to get the full distribution.
    # This performs the full forward pass, giving us a distribution object.
    dist = model.policy.get_distribution(obs)

    # For a Categorical distribution, you can access logits as follows.
    # (Adjust attribute access if your distribution type differs.)
    logits = dist.distribution.logits

    # Compute cross entropy loss.
    loss = F.cross_entropy(logits, gt_act)

    # Backward pass to compute gradients.
    loss.backward()

    # Collect gradients for parameters in the policy network (exclude value network parameters).
    grad_dict = {}
    for name, param in model.policy.named_parameters():
        if param.requires_grad and param.grad is not None:
            # if ("value_net" not in name) and ("mlp_extractor.value_net" not in name):
                grad_dict[name] = param.grad.detach().clone()
    # print(grad_dict)

    return grad_dict

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
    # adjusted_advantages = train_batch.advantages
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

batch_ref = pickle.load(open(args.valid_path, 'rb'))
observations = torch.tensor(np.array([elem['state'] for elem in batch_ref]), dtype=torch.float32, device='cpu')
actions = torch.tensor(np.array([elem['gt_action'] for elem in batch_ref]), dtype=torch.long, device='cpu').unsqueeze(-1)
samples_ref = {'observations': observations, 'actions': actions}

d_global = {}
all_scores = []
for i in tqdm(range(args.begin_step, args.num_updates)):
    model_i = PPO.load(f'./models_{args.env_short}/{args.model_path}/policy_grad_{i}.zip', device="cpu")
    batch_i = pickle.load(open(f'./models_{args.env_short}/{args.model_path}/batch_{i+1}.pkl', 'rb'))
    
    progress = 1-(int(i/320)+1)/(args.total_updates/320)

    grad_ref = compute_ce_policy_gradient(model_i, samples_ref)
    # grad_full = compute_ppo_policy_gradient(model_i, batch_i, progress, normalize_advantages=False)
    grad_i = compute_gradients_for_train_batch(model_i, batch_i, progress)
    scores = [inner_product_gradients(grad, grad_ref).item() for grad in grad_i]
    # pickle.dump(grad_full, open(f'grad_full.pkl', 'wb'))

    all_scores += scores
    update_scores_for_samples(batch_i, scores)
    
os.makedirs(args.output_dir, exist_ok=True)
pickle.dump({
    'all_scores': all_scores,
    'd_global': d_global
}, open(os.path.join(args.output_dir, 'scores.pkl'), 'wb'))

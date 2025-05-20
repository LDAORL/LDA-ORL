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

def compute_action_prob(model, sample):
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

    dist = model.policy.get_distribution(obs)
    logits = dist.distribution.logits
    return logits
   

batch_ref = pickle.load(open(args.valid_path, 'rb'))
observations = torch.tensor(np.array([elem['state'] for elem in batch_ref]), dtype=torch.float32, device='cpu')
actions = torch.tensor(np.array([elem['gt_action'] for elem in batch_ref]), dtype=torch.long, device='cpu').unsqueeze(-1)
samples_ref = {'observations': observations, 'actions': actions}

all_scores = []
logits = []
os.makedirs(args.output_dir, exist_ok=True)
for i in tqdm(range(args.begin_step, args.begin_step+1)):
    model_i = PPO.load(f'./models_{args.env_short}/{args.model_path}/policy_grad_{i}.zip', device="cpu")
    logits = compute_action_prob(model_i, samples_ref)
    # print(logits)
    torch.save(logits.detach().cpu(), 
               os.path.join(args.output_dir, f'logits.pt'))
    
# pickle.dump({
#     'all_scores': all_scores,
#     'd_global': d_global
# }, open(os.path.join(args.output_dir, 'scores.pkl'), 'wb'))

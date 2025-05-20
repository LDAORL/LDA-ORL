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
parser.add_argument('--env_short', type=str, default='minigrid')
args = parser.parse_args()


def flatten_ob_act_array(ob, act):
    return tuple(np.concatenate((ob.flatten(), act.flatten())))

def flatten_ob_array(ob):
    return tuple(ob.flatten())

all_states = np.load(f'./models_{args.env_short}/{args.model_path}/all_states.npy')
all_actions = np.load(f'./models_{args.env_short}/{args.model_path}/all_actions.npy')
all_rewards = np.load(f'./models_{args.env_short}/{args.model_path}/all_rewards.npy')
all_episode_starts = np.load(f'./models_{args.env_short}/{args.model_path}/all_episode_starts.npy')

n = len(all_rewards)
all_returns = np.empty_like(all_rewards)

G = 0
gamma = 0.99

for i in reversed(range(n)):
    if i < n - 1 and all_episode_starts[i+1]:
        G = all_rewards[i]
    else:
        G = G * gamma + all_rewards[i]
    all_returns[i] = G

V = {}
Q = {}
for (state, action, ret) in zip(all_states, all_actions, all_returns):
    ob_act = flatten_ob_act_array(state, action)
    ob = flatten_ob_array(state)
    if ob_act in Q:
        Q[ob_act].append(ret)
    else:
        Q[ob_act] = [ret]

    if ob in V:
        V[ob].append(ret)
    else:
        V[ob] = [ret]

V_ave = {k: np.mean(v) for k, v in V.items()}
Q_ave = {k: np.mean(v) for k, v in Q.items()}
A_ave = {}
for ob_act, value in Q_ave.items():
    # print(ob_act)
    ob = ob_act[:-1]
    A_ave[ob_act] = value - V_ave[ob]
    
pickle.dump({
    'Q': Q,
    'V': V,
    'Q_ave': Q_ave,
    'V_ave': V_ave,
    'A_ave': A_ave
}, open(os.path.join(f'./models_{args.env_short}/{args.model_path}', 'advantage_estimate.pkl'), 'wb'))

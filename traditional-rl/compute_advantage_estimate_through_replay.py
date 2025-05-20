import argparse
from stable_baselines3 import PPO
import numpy as np
np.float_ = np.float64
import gymnasium as gym
import pickle
from tqdm import tqdm
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='easy_to_hard_40k_ckpt_batch')
parser.add_argument('--num_trajectories', type=int, default=100)
parser.add_argument('--output_dir', type=str, default='./outputs_debug')
parser.add_argument('--begin_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=500)
parser.add_argument('--score_dir', type=str, required=True)
args = parser.parse_args()

def pickle_load(filename):
    return pickle.load(open(filename, 'rb'))

def set_state(idx, sorted_scores):
    custom_obs = np.empty(4)
    custom_obs[0] = sorted_scores[idx][0][1]
    custom_obs[1] = sorted_scores[idx][0][3]
    custom_obs[2] = sorted_scores[idx][0][4]
    custom_obs[3] = sorted_scores[idx][0][5]
    
    np.array([sorted_scores[idx][0][6] for idx in range(-20, 0)])
    
    env.state = env.unwrapped.state = custom_obs
    return np.array(sorted_scores[idx][0][:6])

def flatten_ob_act_array(ob, act):
    return tuple(np.concatenate((ob.flatten(), act.flatten())))

def flatten_ob_array(ob):
    return tuple(ob.flatten())

env = gym.make("Acrobot-v1", render_mode="rgb_array")
env.reset()

scores = pickle_load(f'{args.score_dir}/scores.pkl')
d_global = scores['d_global']
sorted_scores = sorted(d_global.items(), key=lambda elem: -np.mean(elem[1]))

model = PPO.load(args.model_path, device='cpu')

def collect_trajectories(idx, sorted_scores, num_trajectories):
    record_act = sorted_scores[idx][0][6]
    
    for _ in range(num_trajectories):
        # 1. reset to a specified state
        env.reset()
        obs = set_state(idx, sorted_scores)
        orig_obs = obs.copy()
        
        reward_list = []
        next_action = None

        # 2. start to take action at this state until trajectory ends
        while True:
            action, _ = model.predict(obs, deterministic=False)
            
            if next_action is None:
                next_action = action
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            # print(action, reward)
            reward_list.append(reward)
            if terminated or truncated:
                break
            
        # 3. compute reward of the trajectory
        G = 0
        for x in reversed(reward_list):
            G = G * 0.99 + x
            
        ob_act = flatten_ob_act_array(orig_obs, next_action)
        ob = flatten_ob_array(orig_obs)
            
        # 4. fill in the Q and V dictionaries
        Q[ob_act] = Q.get(ob_act, []) + [G]
        V[ob] = V.get(ob, []) + [G]
        
    # 5. compute ave_Q, ave_V, ave_A
    ob_act_record = flatten_ob_act_array(orig_obs, record_act)
    ob = flatten_ob_array(orig_obs)
    A_ave[ob_act_record] = np.mean(Q[ob_act_record]) - np.mean(V[ob])

Q = {}
V = {}
A_ave = {}
for i in tqdm(range(args.begin_idx, args.end_idx)):
    collect_trajectories(i, sorted_scores, args.num_trajectories)
Q_ave = {k: np.mean(v) for k, v in Q.items()}
V_ave = {k: np.mean(v) for k, v in V.items()}

os.makedirs(args.output_dir, exist_ok=True)
pickle.dump({
    'Q': Q,
    'V': V,
    'Q_ave': Q_ave,
    'V_ave': V_ave,
    'A_ave': A_ave
}, open(osp.join(args.output_dir, f'advantage_estimate_{args.begin_idx}_{args.end_idx}.pkl'), 'wb'))

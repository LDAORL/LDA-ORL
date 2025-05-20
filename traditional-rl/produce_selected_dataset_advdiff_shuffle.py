import argparse
import os
import os.path as osp
import numpy as np
import pickle
import shutil
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_orig_path', type=str, default='', required=True)
parser.add_argument('--model_path', type=str, default='', required=True)
parser.add_argument('--begin', type=int, default=10)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--ed', type=int, default=11)
parser.add_argument('--final_ed', type=int, default=-1)
parser.add_argument('--n', type=int, default=0)
parser.add_argument('--curriculum', type=str, default='hard_to_easy_30k_1_2')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--valid_samples', default='hard', type=str, help='valid samples')
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--drop_percentile', type=float, default=-1)
parser.add_argument('--env_short', type=str, default='minigrid')
args = parser.parse_args()

assert args.ed == args.k + 1, 'k and ed should be consecutive'
if args.final_ed == -1:
    args.final_ed = args.ed


adv_estimates = pickle.load(open(f'./models_{args.env_short}/{args.model_orig_path}/advantage_estimate.pkl', 'rb'))
A_ave = adv_estimates['A_ave']

def flatten_ob_act_tensors(ob, act):
    return tuple(np.concatenate((ob.cpu().numpy().flatten(), act.cpu().numpy().flatten())))

def flatten_ob_act_prob_adv(ob, act, prob, adv):
    return tuple(np.concatenate((ob.cpu().numpy().flatten(), act.cpu().numpy().flatten(), prob.cpu().numpy().flatten(), adv.cpu().numpy().flatten())))

new_model_path = f'{args.curriculum}_advdiff-shuffle_seed-{args.seed}_rollout_{args.begin}-{args.k}-{args.final_ed}_drop_{args.k}-{args.ed}_bottom-{args.drop_percentile}_last-{10-args.n}_wrt_{args.valid_samples}_ckpt_batch'
os.makedirs(f'./models_{args.env_short}/{new_model_path}', exist_ok=True)


act_ob_to_discard = set()
rollout_idx = args.k

scores = []
records_list = []

if 'sign' in args.valid_samples:
    est_A_list = []
    record_A_list = []
    for i in range(320*rollout_idx, 320*rollout_idx+32):
        batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
        for j in range(64):
            obs, act, prob, adv = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j]
            ob_act = flatten_ob_act_tensors(obs, act)
            record = flatten_ob_act_prob_adv(obs, act, prob, adv)

            scores.append(np.abs(adv - A_ave[ob_act]))
            est_A_list.append(A_ave[ob_act])
            record_A_list.append(adv)

            records_list.append(record)
            
    np.savez_compressed('adv_diff.npz', est_A_list=est_A_list, record_A_list=record_A_list)
        
    drop_indices_all = np.where(np.array(est_A_list) * np.array(record_A_list) < 0)[0]
    print('#satisfying', len(drop_indices_all))
    
    if len(drop_indices_all) == 0:
        record_to_drop = set()
        print('nothing to drop')
    else:
        drop_num = int(args.drop_percentile * len(scores) / 100)
        if drop_num+1 < len(np.array(scores)[drop_indices_all]):    
            threshold = np.sort(np.array(scores)[drop_indices_all])[-drop_num-1]
        else:
            threshold = np.sort(np.array(scores)[drop_indices_all])[0] - 1
            print('all satisfy -- drop all!!')
        
        record_to_drop = set()
        for idx in drop_indices_all:
            if scores[idx] > threshold:
                record_to_drop.add(records_list[idx])
    
else:
    for i in range(320*rollout_idx, 320*rollout_idx+32):
        batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
        for j in range(64):
            obs, act, prob, adv = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j]
            ob_act = flatten_ob_act_tensors(obs, act)
            record = flatten_ob_act_prob_adv(obs, act, prob, adv)
            scores.append(np.abs(adv - A_ave[ob_act]))
            records_list.append(record)

    drop_num = int(args.drop_percentile * len(scores) / 100)
    drop_indices = np.argsort(scores)[-drop_num:]
    record_to_drop = set()
    for idx in drop_indices:
        record_to_drop.add(records_list[idx])
    print('#drop_indices among 2048', len(drop_indices))
    
pickle.dump(record_to_drop, open('record_to_drop.pkl', 'wb'))


obs_list, act_list, prob_list, adv_list, val_list, ret_list = [], [], [], [], [], []
for i in range(320*rollout_idx, 320*rollout_idx+32):
    batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
    for j in range(64):
        obs, act, prob, adv, val, ret = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j], batch_i.old_values[j], batch_i.returns[j]
        record = flatten_ob_act_prob_adv(obs, act, prob, adv)
        if not record in record_to_drop:
            obs_list.append(obs)
            act_list.append(act)
            prob_list.append(prob)
            adv_list.append(adv)
            val_list.append(val)
            ret_list.append(ret)

print('len remaining records', len(obs_list))

t = 320*rollout_idx
for _ in range(10):
    indices = np.random.permutation(range(len(obs_list)))
    n_batch = int(len(indices) // 64)
    for j in range(n_batch):
        cur_indices = indices[j*64:(j+1)*64]
        results = {}
        results['observations'] = torch.tensor(np.array([obs_list[i] for i in cur_indices]))
        results['actions'] = torch.tensor(np.array([act_list[i] for i in cur_indices]))
        results['old_values'] = torch.tensor(np.array([val_list[i] for i in cur_indices]))
        results['old_log_prob'] = torch.tensor(np.array([prob_list[i] for i in cur_indices]))
        results['advantages'] = torch.tensor(np.array([adv_list[i] for i in cur_indices]))
        results['advantages'] = (results['advantages'] - results['advantages'].mean()) / (results['advantages'].std() + 1e-8)
        results['returns'] = torch.tensor(np.array([ret_list[i] for i in cur_indices]))
        
        pickle.dump(results, open(osp.join(f'./models_{args.env_short}/', new_model_path, f'batch_{t+1}.pkl'), 'wb'))
        t += 1

            
for i in range(t, args.ed*320):
    results = None
    pickle.dump(results, open(osp.join(f'./models_{args.env_short}/', new_model_path, f'batch_{i+1}.pkl'), 'wb'))
    
print('len(act_ob_to_discard)', len(record_to_drop))
print('model_path', osp.join(f'./models_{args.env_short}/', new_model_path))

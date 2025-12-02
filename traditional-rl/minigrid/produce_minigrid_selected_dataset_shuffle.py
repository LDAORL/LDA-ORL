import argparse
import os
import os.path as osp
import numpy as np
import pickle
import shutil
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='', required=True)
parser.add_argument('--model_path', type=str, default='', required=True)
parser.add_argument('--begin', type=int, default=10)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--ed', type=int, default=11)
parser.add_argument('--final_ed', type=int, default=-1)
parser.add_argument('--n', type=int, default=0)
parser.add_argument('--method', type=str, default='TracIn', choices=['TracIn', 'SGDI', 'random', 'SGDI-advnorm', 'SGDI-fixed', 'TracIn-fixed'])
parser.add_argument('--curriculum', type=str, default='hard_to_easy_30k_1_2')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--valid_samples', default='hard', type=str, help='valid samples')
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--drop_percentile', type=float, default=-1.)
parser.add_argument('--random', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='neg', choices=['neg', 'bottom', 'random', 'neg-bottom', 'neg-random'])
parser.add_argument('--env_short', type=str, default='minigrid')
args = parser.parse_args()

assert args.ed == args.k + 1, 'k and ed should be consecutive'
if args.final_ed == -1:
    args.final_ed = args.ed

def flatten_ob_act_tensors(ob, act):
    return tuple(np.concatenate((ob.cpu().numpy().flatten(), act.cpu().numpy().flatten())))

def flatten_ob_act_prob_adv(ob, act, prob, adv):
    return tuple(np.concatenate((ob.cpu().numpy().flatten(), act.cpu().numpy().flatten(), prob.cpu().numpy().flatten(), adv.cpu().numpy().flatten())))

if args.method == "SGDI":
    file = osp.join(args.folder, 'sgd_influence_scores.pkl')
    results = pickle.load(open(file, 'rb'))
    d_global = results['sgd_influence_scores']

elif args.method in ["TracIn", "TracIn-fixed"]:
    file = osp.join(args.folder, 'scores.pkl')
    results = pickle.load(open(file, 'rb'))
    d_global = results['d_global']
    
elif args.method == 'SGDI-advnorm':
    file = osp.join(args.folder, 'sgd_influence_advnorm_scores.pkl')
    results = pickle.load(open(file, 'rb'))
    d_global = results['sgd_influence_advnorm_scores']
    
elif args.method == 'SGDI-fixed':
    file = osp.join(args.folder, 'sgd_influence_scores.pkl')
    results = pickle.load(open(file, 'rb'))
    d_global = results['sgd_influence_scores']
    
else:
    if not args.method == 'random':
        raise NotImplementedError

if args.mode == 'random':
    new_model_path = f'{args.curriculum}_shuffle-adv_{args.method}_seed-{args.seed}_rollout_{args.begin}-{args.k}-{args.final_ed}_drop_{args.k}-{args.ed}_random-{args.drop_percentile}_last-{10-args.n}_wrt_{args.valid_samples}_ckpt_batch'
elif args.mode == 'neg':
    new_model_path = f'{args.curriculum}_shuffle-adv_{args.method}_seed-{args.seed}_rollout_{args.begin}-{args.k}-{args.final_ed}_drop_{args.k}-{args.ed}_neg--1.0_last-{10-args.n}_wrt_{args.valid_samples}_ckpt_batch'
else:
    if args.random:
        new_model_path = f'{args.curriculum}_shuffle-adv_{args.method}_seed-{args.seed}_rollout_{args.begin}-{args.k}-{args.final_ed}_drop_{args.k}-{args.ed}_bottom-{args.drop_percentile}-random_last-{10-args.n}_wrt_{args.valid_samples}_ckpt_batch'
    else:
        new_model_path = f'{args.curriculum}_shuffle-adv_{args.method}_seed-{args.seed}_rollout_{args.begin}-{args.k}-{args.final_ed}_drop_{args.k}-{args.ed}_{args.mode}-{args.drop_percentile}_last-{10-args.n}_wrt_{args.valid_samples}_ckpt_batch'
os.makedirs(f'./models_{args.env_short}/{new_model_path}', exist_ok=True)


act_ob_to_discard = set()
rollout_idx = args.k

print('mode', args.mode)

if args.mode in ['bottom', 'neg-bottom']:
    scores = []
    for i in range(320*rollout_idx, 320*rollout_idx+32):
        batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
        for j in range(64):
            obs, act, prob, adv = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j]
            ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
            scores.append(np.mean(d_global[ob_act]))

elif args.mode in ['random']:
    ob_act_list = []
    for i in range(320*rollout_idx, 320*rollout_idx+32):
        batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
        for j in range(64):
            obs, act, prob, adv = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j]
            ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
            ob_act_list.append(ob_act)
            
elif args.mode in ['neg-random']:
    scores, ob_act_list = [], []
    for i in range(320*rollout_idx, 320*rollout_idx+32):
        batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
        for j in range(64):
            obs, act, prob, adv = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j]
            ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
            score = np.mean(d_global[ob_act])
            ob_act_list.append(ob_act)
            scores.append(score)

if args.mode in ['neg', 'bottom', 'neg-bottom']:
    if args.drop_percentile != -1:
        if args.mode == 'bottom':
            threshold = np.percentile(scores, args.drop_percentile)
        elif args.mode == 'neg-bottom':
            threshold = sorted(scores)[int(sum(np.array(scores) < 0) * args.drop_percentile * 0.01)]
    else:
        assert args.mode == 'neg'
        threshold = 0
    # threshold = np.percentile(scores, args.drop_percentile) if args.drop_percentile != -1 else 0
    print('threshold', threshold, args.mode, args.drop_percentile)

    for i in range(320*rollout_idx, 320*rollout_idx+32):
        batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
        for j in range(64):
            obs, act, prob, adv = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j]
            # ob_act = flatten_ob_act_tensors(obs, act)
            ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
            if np.mean(d_global[ob_act]) < threshold:
                act_ob_to_discard.add(ob_act)
                # print(i, j)
                
    if args.drop_percentile > 0 and args.random:
        import random
        random.seed(args.seed)
        act_ob_to_discard = set(random.sample(sorted(act_ob_to_discard), 1))
        
elif args.mode == 'random':
    import random
    random.seed(args.seed)
    act_ob_to_discard = set(random.sample(ob_act_list, int(len(ob_act_list)*0.01*args.drop_percentile)))
    
elif args.mode == 'neg-random':
    import random
    random.seed(args.seed)
    scores = np.array(scores)
    threshold = 0
    neg_set = [ob_act_list[i] for i in range(len(ob_act_list)) if scores[i] < threshold]
    print('random sample from the negative set')
    act_ob_to_discard = set(random.sample(neg_set, int(len(neg_set)*0.01*args.drop_percentile)))
      
pickle.dump(act_ob_to_discard, open('act_ob_to_discard_PG.pkl', 'wb'))
# print(np.min(scores), np.max(scores))
# print(np.percentile(scores, 0), np.percentile(scores, 25), np.percentile(scores, 50), np.percentile(scores, 75), np.percentile(scores, 100))
# print('len(act_ob_to_discard)', len(act_ob_to_discard))
# pickle.dump(act_ob_to_discard, open(osp.join('./', 'act_ob_to_discard.pkl'), 'wb'))
# print(act_ob_to_discard)

if args.mode == 'random':
    obs_list, act_list, prob_list, adv_list, val_list, ret_list = [], [], [], [], [], []
    for i in range(320*rollout_idx, 320*rollout_idx+32):
        batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
        for j in range(64):
            obs, act, prob, adv, val, ret = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j], batch_i.old_values[j], batch_i.returns[j]
            ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
            if np.random.rand() < 1-args.drop_percentile*0.01:
                obs_list.append(obs)
                act_list.append(act)
                prob_list.append(prob)
                adv_list.append(adv)
                val_list.append(val)
                ret_list.append(ret)
    
else:
    obs_list, act_list, prob_list, adv_list, val_list, ret_list = [], [], [], [], [], []
    for i in range(320*rollout_idx, 320*rollout_idx+32):
        batch_i = pickle.load(open(osp.join(f'models_{args.env_short}/{args.model_path}', f'batch_{i+1}.pkl'), 'rb'))
        for j in range(64):
            obs, act, prob, adv, val, ret = batch_i.observations[j], batch_i.actions[j], batch_i.old_log_prob[j], batch_i.advantages[j], batch_i.old_values[j], batch_i.returns[j]
            ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
            if not ob_act in act_ob_to_discard:
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
    
if not args.random:
    print('len(act_ob_to_discard)', len(act_ob_to_discard))
print('model_path', osp.join(f'./models_{args.env_short}/', new_model_path))

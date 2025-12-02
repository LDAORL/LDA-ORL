import argparse
import pickle
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='easy_to_hard_40k_ckpt_batch')
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--ed', type=int, default=1)
parser.add_argument('--env_short', type=str, default='minigrid')
parser.add_argument('--n_updates', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=10)
args = parser.parse_args()

for i in range(args.k, args.ed):
    obs_list = []
    act_list = []
    adv_list = []
    prob_list = []
    ret_list = []
    for j in range(i*args.n_updates*args.n_epochs, i*args.n_updates*args.n_epochs+args.n_updates):
        data = pickle.load(open(f'{args.model_path}/batch_{j+1}.pkl', 'rb'))
        obs_list.append(data.observations)
        act_list.append(data.actions)
        adv_list.append(data.advantages)
        prob_list.append(data.old_log_prob)
        ret_list.append(data.returns)
    obs = torch.cat(obs_list, dim=0).unsqueeze(1)
    act = torch.cat(act_list, dim=0).unsqueeze(1)
    adv = torch.cat(adv_list, dim=0).unsqueeze(1)
    prob = torch.cat(prob_list, dim=0).unsqueeze(1)
    ret = torch.cat(ret_list, dim=0).unsqueeze(1)
    
    buffer_save = {
        'observations': obs,
        'actions': act,
        'advantages': adv,
        'log_probs': prob,
        'returns': ret,
    }
    
    print(f"Buffer {i} shape: {obs.shape}, {act.shape}, {adv.shape}, {prob.shape}")
    
    with open(f'{args.model_path}/buffer_{i}.pkl', 'wb') as f:
        pickle.dump(buffer_save, f)
    print(f"Saved buffer_{i}")

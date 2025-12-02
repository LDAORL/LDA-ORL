set -x

policy_folder=$1
beg_idx=$2
end_idx=$3

for i in $(seq $beg_idx $end_idx)
do
    step=$((i*320))
    python test_minigrid_act3_return.py \
        --n_actions 7 \
        --env_name MiniGrid-Empty-8x8-v0 \
        --policy_path models_minigrid/${policy_folder}/policy_grad_${step}.zip \
        --num_episodes 1000
done

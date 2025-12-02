set -x

policy_folder=$1
beg_idx=$2
end_idx=$3

for i in $(seq $beg_idx $end_idx)
do
    step=$((i*320))
    python test_lunarlander_return.py \
        --policy_path models_lunarlander/${policy_folder}/policy_grad_${step}.zip \
        --num_episodes 1000
done

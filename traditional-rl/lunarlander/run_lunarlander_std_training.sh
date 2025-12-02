set -x
seed=$1
python train_lunarlander_with_callbacks.py \
    --learning_rate 5e-3 \
    --total_timesteps 307200 \
    --save_path ./models_lunarlander/lunarlander_lr-5e-3_step-310k_seed-${seed} \
    --seed ${seed}

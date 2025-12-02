set -x
seed=$1
python train_lunarlander_with_callbacks.py \
    --learning_rate 1e-3 \
    --total_timesteps 307200 \
    --optimizer_class Adam \
    --save_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_seed-${seed} \
    --seed ${seed}

    # --learning_rate 3e-4 \
    
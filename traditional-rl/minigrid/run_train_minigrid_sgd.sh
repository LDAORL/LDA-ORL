set -x
seed=$1
python train_minigrid_with_callbacks.py \
    --learning_rate 5e-3 \
    --total_timesteps 81920 \
    --features_dim 64 \
    --seed ${seed} \
    --save_path ./models_minigrid/minigrid_88_sgd_lr-5e-3_feat-64_step-80k_seed-${seed} \
    --env_name MiniGrid-Empty-8x8-v0 \
    --project_name minigrid_88_rl \
    --optimizer_class SGD \
    --run_name minigrid_88_sgd_lr-5e-3_feat-64_step-80k_seed-${seed} \
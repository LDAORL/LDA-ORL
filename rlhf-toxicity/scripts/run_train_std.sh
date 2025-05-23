set -x
accelerate launch --main_process_port=29522 \
    --num_machines 1  \
    --num_processes 1 \
    train_rlhf.py --log_with=wandb \
    --model_name="EleutherAI/gpt-neo-2.7B" \
    --dataset_name="allenai/real-toxicity-prompts" \
    --reward_model_name="facebook/roberta-hate-speech-dynabench-r4-target" \
    --adafactor=False \
    --save_freq=1 \
    --batch_size=256 \
    --tracin_batch_size=256 \
    --mini_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=4 \
    --seed=22 \
    --max_length=30 \
    --gen_bsize=256 \
    --val_size=1024 \
    --learning_rate=1e-5 \
    --early_stopping=False \
    --output_dir=output_tox_std_2.7b_bfloat16_kl-0.04_mbs-1_seed-22 \
    --init_kl_coef=0.04 --steps=200 \
    --min_length=20 \
    --wandb_project="ppo-detox" \
    --run_name="std-2.7b-bfloat16_kl-0.04_mbs-1_seed-22" \
    --gen_data_dir="gen_tox_all_samples_std_2.7b_bfloat16_kl-0.04_mbs-1_seed-22" \

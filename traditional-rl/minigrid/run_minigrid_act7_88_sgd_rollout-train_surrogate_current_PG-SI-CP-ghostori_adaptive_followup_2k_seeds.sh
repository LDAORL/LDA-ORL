set -x

begin=$1
k=$2
pk=$(($k-1))
ed=$(($k+1))
k_idx=$(($k*320))
ed_idx=$(($ed*320))
mode=$3
x=$4
valid=$5
gae_lambda=$6
n_actions=$7
seed=${8:-42}

start_time=$(date +%s)

printf "\n*** exploring using last policy\n"

python train_minigrid_act3_with_given_states.py \
  --env_name MiniGrid-Empty-8x8-v0 \
  --n_actions ${n_actions} \
  --total_timesteps 81920 \
  --learning_rate 5e-3 \
  --optimizer_class SGD \
  --features_dim 64 \
  --seed ${seed} \
  --initial_policy ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_TracIn_${begin}-${pk}-${k}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
  --rollout_begin_idx ${k} \
  --total_rollouts ${ed} \
  --save_path ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}

time1=$(date +%s)
elapsed=$((time1 - start_time))
printf "\nTime elapsed after exploration: %d seconds\n" $elapsed

printf "\n*** turn training rollout into validation rollout\n"

python merge_training_buffers.py \
  --model_path ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --k ${k} \
  --ed ${ed}

time2=$(date +%s)
elapsed=$((time2 - time1))
printf "\nTime elapsed after rollout collection: %d seconds\n" $elapsed

# printf "\n*** test the policy obtained on exploration data\n"

# python test_minigrid_act3_return.py \
#   --env_name MiniGrid-Empty-8x8-v0 \
#   --n_actions ${n_actions} \
#   --policy_path ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
#   --num_episodes 1000

# python test_minigrid_surrogate_PG-IS_loss.py \
#   --model_path ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
#   --ref_policy ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_TracIn_${begin}-${pk}-${k}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
#   --rollout_file ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl

time3=$(date +%s)
elapsed=$((time3 - time2))
printf "\nTime elapsed after policy testing: %d seconds\n" $elapsed

printf "\n*** computing TracIn using collected rollout on the collected exploration data\n"

python compute_TracIn_surrogate_ghost_with_CNN.py \
  --rollout_file ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl \
  --model_path minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --ref_policy ${k_idx} \
  --begin_step ${k_idx} \
  --num_updates ${ed_idx} \
  --total_updates 12800 \
  --output_dir outputs_TracIn-CP-ghost_minigrid_act-7_88_seed-${seed}_sgd_lr-5e-3_step-80k_valid-${valid}_begin-${begin}-updates-${k_idx}-${ed_idx}_${mode}-${x}

time4=$(date +%s)
elapsed=$((time4 - time3))
printf "\nTime elapsed after TracIn computation: %d seconds\n" $elapsed

printf "\n*** perform dataset selection on the collected exploration data\n"

python produce_minigrid_selected_dataset_shuffle.py \
  --folder outputs_TracIn-CP-ghost_minigrid_act-7_88_seed-${seed}_sgd_lr-5e-3_step-80k_valid-${valid}_begin-${begin}-updates-${k_idx}-${ed_idx}_${mode}-${x} \
  --model_path minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --begin ${begin} --k ${k} --ed ${ed} \
  --method TracIn \
  --curriculum minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k \
  --seed ${seed} \
  --valid_samples ${valid} \
  --drop_percentile ${x} \
  --mode ${mode}

printf "\n*** train on the selected dataset\n"

python train_minigrid_act3_with_given_batches.py \
  --env_name MiniGrid-Empty-8x8-v0 \
  --n_actions ${n_actions} \
  --seed ${seed} \
  --initial_policy ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_TracIn_${begin}-${pk}-${k}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
  --save_path ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_TracIn_${begin}-${k}-${ed}_${mode}-${x}_valid-${valid} \
  --run_name train_with_given_states \
  --batches_dir ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_shuffle-adv_TracIn_seed-${seed}_rollout_${begin}-${k}-${ed}_drop_${k}-${ed}_${mode}-${x}_last-10_wrt_${valid}_ckpt_batch \
  --batch_begin_idx ${k_idx} \
  --n_batches ${ed_idx} \
  --total_updates 12800 \
  --batch_size 64 \
  --learning_rate 5e-3 \
  --optimizer_class SGD \
  --features_dim 64

time5=$(date +%s)
elapsed=$((time5 - time4))
printf "\nTime elapsed after dataset selection and training: %d seconds\n" $elapsed

printf "\n*** test the final policy\n"

python test_minigrid_surrogate_PG-IS_loss.py \
  --model_path ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_TracIn_${begin}-${k}-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
  --ref_policy ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_TracIn_${begin}-${pk}-${k}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
  --rollout_file ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl

python test_minigrid_act3_return.py \
  --env_name MiniGrid-Empty-8x8-v0 \
  --n_actions ${n_actions} \
  --policy_path ./models_minigrid/minigrid_act-7_88_sgd_lr-5e-3_feat-64_step-80k_TracIn_${begin}-${k}-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
  --num_episodes 1000

end_time=$(date +%s)
elapsed=$((end_time - time5))
printf "\nTime elapsed after final testing: %d seconds\n" $elapsed
total_elapsed=$((end_time - start_time))
printf "\nTotal execution time: %d seconds\n" $total_elapsed
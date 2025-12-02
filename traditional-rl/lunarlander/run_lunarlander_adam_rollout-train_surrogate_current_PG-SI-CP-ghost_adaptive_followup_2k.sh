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
seed=${7:-42}

start_time=$(date +%s)

printf "\n*** exploring using last policy\n"

python train_lunarlander_with_given_states.py \
  --total_timesteps 307200 \
  --optimizer_class Adam \
  --learning_rate 1e-3 \
  --seed ${seed} \
  --initial_policy ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${begin}-${pk}-${k}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
  --rollout_begin_idx ${k} \
  --total_rollouts ${ed} \
  --save_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}

time1=$(date +%s)
elapsed=$((time1 - start_time))
printf "\nTime elapsed after exploration: %d seconds\n" $elapsed

printf "\n*** turn training rollout into validation rollout\n"

python merge_training_buffers.py \
  --model_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --env_short lunarlander \
  --k ${k} \
  --ed ${ed}

time2=$(date +%s)
elapsed=$((time2 - time1))
printf "\nTime elapsed after rollout collection: %d seconds\n" $elapsed

# printf "\n*** test the policy obtained on exploration data\n"

# python test_lunarlander_return.py \
#   --policy_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
#   --num_episodes 1000

# python test_minigrid_surrogate_PG-IS_loss.py \
#   --model_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
#   --ref_policy ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${begin}-${pk}-${k}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
#   --rollout_file ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl

# time3=$(date +%s)
# elapsed=$((time3 - time2))
# printf "\nTime elapsed after policy testing: %d seconds\n" $elapsed

printf "\n*** computing TracIn using collected rollout on the collected exploration data\n"

python compute_TracIn_surrogate_ghost.py \
  --env_short lunarlander \
  --rollout_file ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl \
  --model_path lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --ref_policy ${k_idx} \
  --begin_step ${k_idx} \
  --num_updates ${ed_idx} \
  --total_updates 48000 \
  --output_dir outputs_TracIn_lunarlander_seed-${seed}_adam_lr-1e-3_step-310k_valid-${valid}_begin-${begin}-updates-${k_idx}-${ed_idx}_${mode}-${x}

time4=$(date +%s)
elapsed=$((time4 - time2))
printf "\nTime elapsed after TracIn computation: %d seconds\n" $elapsed

printf "\n*** perform dataset selection on the collected exploration data\n"

python produce_minigrid_selected_dataset_shuffle.py \
  --env_short lunarlander \
  --folder outputs_TracIn_lunarlander_seed-${seed}_adam_lr-1e-3_step-310k_valid-${valid}_begin-${begin}-updates-${k_idx}-${ed_idx}_${mode}-${x} \
  --model_path lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --begin ${begin} --k ${k} --ed ${ed} \
  --method TracIn \
  --curriculum lunarlander_adam_lr-1e-3_step-310k \
  --seed ${seed} \
  --valid_samples ${valid} \
  --drop_percentile ${x} \
  --mode ${mode}

printf "\n*** train on the selected dataset\n"

python train_lunarlander_with_given_batches.py \
  --seed ${seed} \
  --initial_policy ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${begin}-${pk}-${k}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
  --save_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${begin}-${k}-${ed}_${mode}-${x}_valid-${valid} \
  --run_name train_with_given_states \
  --batches_dir ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_shuffle-adv_TracIn_seed-${seed}_rollout_${begin}-${k}-${ed}_drop_${k}-${ed}_${mode}-${x}_last-10_wrt_${valid}_ckpt_batch \
  --batch_begin_idx ${k_idx} \
  --n_batches ${ed_idx} \
  --total_updates 48000 \
  --batch_size 64 \
  --optimizer_class Adam \
  --learning_rate 1e-3

time5=$(date +%s)
elapsed=$((time5 - time4))
printf "\nTime elapsed after dataset selection and training: %d seconds\n" $elapsed

printf "\n*** test the final policy\n"

# python test_minigrid_surrogate_PG-IS_loss.py \
#   --model_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${begin}-${k}-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
#   --ref_policy ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${begin}-${pk}-${k}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
#   --rollout_file ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${begin}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl

python test_lunarlander_return.py \
  --policy_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${begin}-${k}-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
  --num_episodes 1000

end_time=$(date +%s)
elapsed=$((end_time - time5))
printf "\nTime elapsed after final testing: %d seconds\n" $elapsed
total_elapsed=$((end_time - start_time))
printf "\nTotal execution time: %d seconds\n" $total_elapsed
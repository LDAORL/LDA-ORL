set -x

k=$1
pk=$(($k-1))
ed=$(($k+1))
k_idx=$(($k*320))
ed_idx=$(($ed*320))
mode=$2
x=$3
valid=$4
gae_lambda=$5
seed=${6:-42}
if [ ${seed} -eq 182 ]; then
  suffix=''
else
  suffix="_seed-${seed}"
fi

printf "\n*** exploring using last policy\n"

python train_lunarlander_with_given_states.py \
  --total_timesteps 307200 \
  --optimizer_class Adam \
  --learning_rate 1e-3 \
  --seed ${seed} \
  --initial_policy ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k${suffix}/policy_grad_${k_idx}.zip \
  --rollout_begin_idx ${k} \
  --total_rollouts ${ed} \
  --save_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}

printf "\n*** turn training rollout into validation rollout\n"

python merge_training_buffers.py \
  --model_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --env_short lunarlander \
  --k ${k} \
  --ed ${ed}

# printf "\n*** test the policy obtained on exploration data\n"

# python test_lunarlander_return.py \
#   --policy_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
#   --num_episodes 1000

# python test_minigrid_surrogate_PG-IS_loss.py \
#   --model_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
#   --ref_policy ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
#   --rollout_file ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl

printf "\n*** computing TracIn using collected rollout on the collected exploration data\n"

python compute_TracIn_surrogate_ghost.py \
  --env_short lunarlander \
  --rollout_file ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl \
  --model_path lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --ref_policy ${k_idx} \
  --begin_step ${k_idx} \
  --num_updates ${ed_idx} \
  --total_updates 48000 \
  --output_dir outputs_TracIn_lunarlander_seed-${seed}_adam_lr-1e-3_step-310k_valid-${valid}_begin-${k}-updates-${k_idx}-${ed_idx}_${mode}-${x}

printf "\n*** perform dataset selection on the collected exploration data\n"

python produce_minigrid_selected_dataset_shuffle.py \
  --env_short lunarlander \
  --folder outputs_TracIn_lunarlander_seed-${seed}_adam_lr-1e-3_step-310k_valid-${valid}_begin-${k}-updates-${k_idx}-${ed_idx}_${mode}-${x} \
  --model_path lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid} \
  --begin ${k} \
  --k ${k} --ed ${ed} \
  --method TracIn \
  --curriculum lunarlander_adam_lr-1e-3_step-310k \
  --seed ${seed} \
  --valid_samples ${valid} \
  --drop_percentile ${x} \
  --mode ${mode}

printf "\n*** train on the selected dataset\n"

python train_lunarlander_with_given_batches.py \
  --seed ${seed} \
  --initial_policy ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k${suffix}/policy_grad_${k_idx}.zip \
  --save_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${k}-${k}-${ed}_${mode}-${x}_valid-${valid} \
  --run_name train_with_given_states \
  --batches_dir ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_shuffle-adv_TracIn_seed-${seed}_rollout_${k}-${k}-${ed}_drop_${k}-${ed}_${mode}-${x}_last-10_wrt_${valid}_ckpt_batch \
  --batch_begin_idx ${k_idx} \
  --n_batches ${ed_idx} \
  --total_updates 48000 \
  --batch_size 64 \
  --optimizer_class Adam \
  --learning_rate 1e-3

printf "\n*** test the final policy\n"

# python test_minigrid_surrogate_PG-IS_loss.py \
#   --model_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${k}-${k}-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
#   --ref_policy ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${k_idx}.zip \
#   --rollout_file ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-${k}-from-${k}-to-${ed}_${mode}-${x}_valid-${valid}/buffer_${k}.pkl

python test_lunarlander_return.py \
  --policy_path ./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_${k}-${k}-${ed}_${mode}-${x}_valid-${valid}/policy_grad_${ed_idx}.zip \
  --num_episodes 1000



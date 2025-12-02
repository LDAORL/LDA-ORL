set -x
seed=$1
begin_idx=${2:-0}
end_idx=${3:-150}
bash run_test_lunarlander_policy.sh lunarlander_lr-5e-3_step-310k_seed-${seed} ${begin_idx} ${end_idx} \
  > results_lunarlander/result_lunarlander_lr-5e-3_step-310k_seed-${seed}_${begin_idx}-${end_idx}.txt

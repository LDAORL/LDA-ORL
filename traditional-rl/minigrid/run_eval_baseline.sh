set -x
seed=$1
cpus=$2
taskset -c ${cpus} bash run_test_minigrid_policy_act7.sh minigrid_88_sgd_lr-5e-3_feat-64_step-80k_seed-${seed} 0 20 \
  > results_minigrid/result_minigrid_88_sgd_lr-5e-3_feat-64_step-80k_seed-${seed}.txt;\

set -x
seed=$1
pct=$2
cpus=$3
taskset -c ${cpus} bash run_lunarlander_adam_rollout-train_surrogate_current_PG-SI-CP-ghost_adaptive_2k.sh 0 neg-bottom ${pct} PG-SI-CP-ghost-2k-lambda-1-rollout-train-seed-${seed} 1 ${seed} >> \
    results_lunarlander/result_lunarlander_adam_rollout_train_surrogate_PG-SI-CP-ghost_adaptive_rol-0-0-1_neg-bottom-${pct}_gae-lambda-1_2k_lambda_seed-${seed}.txt;\
for i in {1..99}
do
    j=$((i+1))
    taskset -c ${cpus} bash run_lunarlander_adam_rollout-train_surrogate_current_PG-SI-CP-ghost_adaptive_followup_2k.sh 0 ${i} neg-bottom ${pct} PG-SI-CP-ghost-2k-lambda-1-rollout-train-seed-${seed} 1 ${seed} > \
        results_lunarlander/result_lunarlander_adam_rollout_train_surrogate_PG-SI-CP-ghost_adaptive_rol-0-${i}-${j}_neg-bottom-${pct}_gae-lambda-1_2k_lambda_seed-${seed}.txt;\
done

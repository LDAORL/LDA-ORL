- standard training w/ adam

```bash
# Standard training
taskset -c 0-7 bash run_lunarlander_adam_training.sh 7270
# Eval for standard training
taskset -c 0-7 bash run_lunarlander_adam_test.sh 7270 0 100
```

- standard training w/ sgd

```bash
# Standard training
taskset -c 0-7 bash run_lunarlander_std_training.sh 7554
# Eval for standard training
taskset -c 0-7 bash run_lunarlander_std_test.sh 7554 0 100
```

- IIF w/ adam

```bash
bash run_iif_lunarlander_adam.sh 7270 12.5 0-7
```

- IIF w/ sgd

```bash
bash run_iif_lunarlander_sgd.sh 7554 12.5 0-7
```

# Experiments in traditional RL environments

This code repository contains the implementation of data influence calculation in our local data attribution framework and experiments in traditional RL environments in Gymnasium, using Iterative Influence-Based Filtering (IIF) to improve training.

We supply experiments in two environments: MiniGrid and LunarLander. We provide scripts for IIF training, standard training, and evaluation.

## Setup

Set up a conda environment with Python 3.11, and then

1. Installation

```bash
pip install -r requirements.txt
```

2. Replace the ``stable_baselines3`` package in your environment.

```bash
cp -r stable_baselines3 $CONDA_PREFIX/lib/python3.11/site-packages/
```

## Environments

### MiniGrid

**Standard training + eval**
```bash
taskset -c 0-7 bash run_train_minigrid_adam.sh [seed]
taskset -c 0-7 bash run_eval_baseline.sh [seed] 0-7
```

**IIF training + eval**
```bash
bash run_iif_minigrid.sh [seed] 12.5 0-7
```

### LunarLander

**Standard training (w/ Adam) + eval**

```bash
taskset -c 0-7 bash run_lunarlander_adam_training.sh [seed]
taskset -c 0-7 bash run_lunarlander_adam_test.sh [seed] 0 100
```

**IIF training (w/ Adam) + eval**

```bash
bash run_iif_lunarlander_adam.sh [seed] 12.5 0-7
```

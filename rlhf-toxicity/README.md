## RLHF for toxicity mitigation

This code repository is adapted from https://github.com/PrasannS/rlhf-length-biases/tree/master.

### Setup

Set up a conda environment with Python 3.10, and then

1. Installation

```bash
pip install -r requirements.txt
cd rlhf_utils
pip install -e .
cd ..
```

2. Replace ``ppo_trainer.py`` and ``ppo_config.py`` in the ``trl`` package in your environment.

```bash
cp scripts/ppo_trainer.py $CONDA_PREFIX/lib/python3.10/site-packages/trl/trainer/ppo_trainer.py
cp scripts/ppo_config.py $CONDA_PREFIX/lib/python3.10/site-packages/trl/trainer/ppo_config.py
```

### Run the training script

First enter the ``scripts`` directory, and then run the training scripts.

- Standard training:
```bash
bash run_train_std.sh
```

- Training with IIF
```bash
bash run_train_iif.sh
```
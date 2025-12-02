# A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning


This repository contains the official implementation of  
**A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning** (NeurIPS 2025 ${\textsf{\color{red}oral}}$), the first framework of data attribution for online reinforcement learning.
We also propose Iterative Influence-Based Filtering (IIF), an algorithm that improves sample efficiency, computational efficiency, and final returns for online RL. 

Our paper can be accessed [here](https://arxiv.org/abs/2510.12345).

<img src="figs/diagram.png" width="700">

## Repository Structure

This repository is organized as follows:

- `traditional-rl/`: contains data influence calculation in our local data attribution framework and experiments in traditional RL environments in Gymnasium, using IIF to improve training.
- `rlhf-toxicity/`: contains experiment on using IIF to improve the performance of RLHF in the task of LLM detoxification.

## Citation

```bibtex
@inproceedings{hu2025snapshot,
    title={A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning},
    author={Hu, Yuzheng and Wu, Fan and Ye, Haotian and Forsyth, David and Zou, James and Jiang, Nan and Ma, Jiaqi W and Zhao, Han},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=sYK4yPDuT1}
}
```
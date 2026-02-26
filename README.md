# CaRe-BN
[[`📕 arXiv`](https://arxiv.org/pdf/2509.23791)]  [[`💬 OpenReview`](https://openreview.net/forum?id=AaZVrbElhC)]

Official code release for the **ICLR 2026** paper 👇
### CaRe-BN: Precise Moving Statistics for Stabilizing Spiking Neural Networks in Reinforcement Learning
Zijie Xu, Xinyu Shi, Yiting Dong, Zihan Huang, Zhaofei Yu

## Setup
Execute the following commands to set up a conda environment to run experiments
```
conda env create -f environment.yml -n carebn
```


## Running Experiments
Experiments can be run by calling:
```
python main.py --env Ant-v4  --spiking_neurons LIF 
```

The environment "--env" can be "Ant-v4", "HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", and "InvertedDoublePendulum-v4". The spiking neurons "--spiking_neurons" can be "LIF", "CLIF", and "ANN". To test the vanilla spiking actor network, set "--BN" to "No" or "Vanilla". Hyper-parameters can be modified with different arguments to main.py.

## Citing This

To cite our paper and/or this repository in publications:

```bibtex
@inproceedings{
xu2026carebn,
title={CaRe-{BN}: Precise Moving Statistics for Stabilizing Spiking Neural Networks in Reinforcement Learning},
author={Zijie Xu and Xinyu Shi and Yiting Dong and Zihan Huang and Zhaofei Yu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=AaZVrbElhC}
}
```

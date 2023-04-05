# Cell Attention Networks

This repository contains the official code implementation for the paper 
[Cell Attention Networks](https://arxiv.org/abs/2209.08179).

Cell Attention Networks propose a novel message-passing scheme for graph neural networks (GNNs) that lifts node feature vectors into a higher-dimensional space called Cellular Attention Networks. The information exchange between edges is weighted by learnable attention coefficients, which enhances the model's expressiveness and generalization.

![Cell Attention Network Diagram](./images/can_diagram.png)

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Examples](#examples)
- [References](#references)
- [Citation](#citation)

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- torch-geometric 2.0+
- numpy 1.20+
- tqdm 4.62+

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```


## Running experiments on TUDatasets




```commandline
python ./exp/run_tu.py
```


## Running all results on TUDatasets

```commandline
sh ./exp/run_table.sh
```

## Citation
If you find this work useful, please consider citing the paper:

```
@misc{giusti2022cell,
      title={Cell Attention Networks}, 
      author={Lorenzo Giusti and Claudio Battiloro and Lucia Testa and Paolo Di Lorenzo and Stefania Sardellitti and Sergio Barbarossa},
      year={2022},
      eprint={2209.08179},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
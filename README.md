<img src="https://render.githubusercontent.com/render/math?math=\text{G}^2\text{GNN}" style="width:200px;">
This repository is an $1^{st}$ official PyTorch(Geometric) implementation of GNN-based machine learning models for handling imbalanced graph classification.

See more details for the paper ['Imbalanced Graph Classification via Graph-of-Graph Neural Networks'](https://dl.acm.org/doi/10.1145/3511808.3557356)

**If you use this code, please consider citing:**
```linux
@inproceedings{10.1145/3511808.3557356,
author = {Wang, Yu and Zhao, Yuying and Shah, Neil and Derr, Tyler},
title = {Imbalanced Graph Classification via Graph-of-Graph Neural Networks},
year = {2022},
series = {CIKM '22}
}
```

## Requirements
* PyTorch 1.11.0+cu113
* PyTorch Geometric 2.0.4
* torch-scatter 2.0.9
* torch-sparse 0.6.15
* torch-cluster 1.6.0

Note that the **version of the PyTorch Geometric/scatter/sparse/cluster used here is not the latest 2.1.0**, which can be intalled via pip as follows:
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric==2.0.4
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

## Implemented GNN models
* [ICLR 2019] **GIN-How Powerful Are Graph Neural Networks?** [[paper]](https://arxiv.org/pdf/1810.00826.pdf)
* [ICLR 2020] **InfoGraph-InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization** [[paper]](https://arxiv.org/abs/1908.01000)
* [Neurips 2020] **GraphCL-Graph Contrastive Learning with Augmentations** [[paper]](https://arxiv.org/abs/2010.13902)

## Implemented strategies for handling imbalance issue in graph classification
* [ICLR 2019] **GIN-How Powerful Are Graph Neural Networks?** [[paper]](https://arxiv.org/pdf/1810.00826.pdf)
* [ICLR 2020] **InfoGraph-InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization** [[paper]](https://arxiv.org/abs/1908.01000)
* [Neurips 2020] **GraphCL-Graph Contrastive Learning with Augmentations** [[paper]](https://arxiv.org/abs/2010.13902)


## Run


* To reproduce the performance comparison and the ablation study in the following Table and the Figure, run
```linux
bash run_{dataset}.sh
```

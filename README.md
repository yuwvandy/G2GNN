<img src="https://render.githubusercontent.com/render/math?math=\text{G}^2\text{GNN}" style="width:200px;">
This repository is an $1^{st}$ official PyTorch(Geometric) implementation of GNN-based machine learning models for handling imbalanced graph classification.

See more details for the paper ['Imbalanced Graph Classification via Graph-of-Graph Neural Networks'](https://dl.acm.org/doi/10.1145/3511808.3557356)

**If you use this code, please consider citing:**
```linux
@inproceedings{wang2022imbalance,
author = {Wang, Yu and Zhao, Yuying and Shah, Neil and Derr, Tyler},
title = {Imbalanced Graph Classification via Graph-of-Graph Neural Networks},
year = {2022},
booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
pages = {2067–2076},
numpages = {10},
series = {CIKM '22}
}
```

## Requirements
* PyTorch 1.11.0+cu113
* PyTorch Geometric 2.0.4
* torch-scatter 2.0.9
* torch-sparse 0.6.15
* torch-cluster 1.6.0

Note that the **version of the PyTorch Geometric/scatter/sparse/cluster used here is not the very latest one**. The current used versions can be intalled via:
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
* [ICLR 2020] **InfoGraph-InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization** [[paper]](https://arxiv.org/abs/1908.01000) - incoming soon!
* [Neurips 2020] **GraphCL-Graph Contrastive Learning with Augmentations** [[paper]](https://arxiv.org/abs/2010.13902) - incoming soon!

## Implemented strategies for handling imbalance issue in graph classification
* [ICML 1997] **Upsampling: Addressing the curse of imbalanced training sets: one-sided selection** [[paper]](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/kubat97addressing.pdf)
* [IJCNN 2012] **Reweight: Sampling + reweighting: Boosting the performance of AdaBoost on imbalanced datasets** [[paper]](https://ieeexplore.ieee.org/document/6252738)
* [JAIR 2002] **SMOTE: SMOTE: Synthetic Minority Over-sampling Technique** [[paper]](https://arxiv.org/pdf/1106.1813.pdf)
* [CIKM 2022] **G2GNN: Imbalanced Graph Classification via Graph-of-Graph Neural Networks** [[paper]](https://dl.acm.org/doi/10.1145/3511808.3557356)
* [CIKM 2022] **Remove edges: Imbalanced Graph Classification via Graph-of-Graph Neural Networks** [[paper]](https://dl.acm.org/doi/10.1145/3511808.3557356)
* [CIKM 2022] **Masking nodes: Imbalanced Graph Classification via Graph-of-Graph Neural Networks** [[paper]](https://dl.acm.org/doi/10.1145/3511808.3557356)


## Run
Note that compared to the previous verion of this repository, we move the K-nearest neighbor search in topological space into the batch-processing, which hence can be speed up due to parallel preparation. Furthermore, to solve the undeterministic issue, we replace the original [scatter-based message-passing/pooling](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html) with sparse-matrix multiplication-based message-passing and [segment_csr-based pooling](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html), see more details [[here]](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html)

```linux
bash run_{dataset}.sh
```

from utils import *
from dataset import *

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath, RandomWalk, WeisfeilerLehman, GraphletSampling, SubgraphMatching
import torch_geometric.transforms as T

dataname = 'MUTAG'

dataset, n_feat, n_class, mapping = get_TUDataset(dataname, pre_transform=T.ToSparseTensor())
print(dataset.data, (dataset.data.y == 0).sum(), (dataset.data.y == 1).sum())
kernel_name = ['SP', 'RW', 'SM', 'WL', 'GS']
kernel = [ShortestPath, RandomWalk, SubgraphMatching,
          WeisfeilerLehman, GraphletSampling]
k = 11

# ====== compute the kernel and select the top k
dataset = fetch_dataset(dataname, verbose=False)
G = dataset.data
y = torch.tensor(dataset.target)

for i in range(len(kernel_name)):
    if(kernel_name[i] == 'SP'):
        kernel_simi = torch.tensor(
            kernel[i](normalize=True, with_labels=True).fit_transform(G))
    else:
        kernel_simi = torch.tensor(kernel[i](normalize=True).fit_transform(G))

    for k in range(10):
        kernel_idx = torch.topk(kernel_simi, k=(
            k + 2), dim=1, largest=True)[1][:, 1:]
        knn_edge_index = construct_knn(kernel_idx)

        edge_homo, node_homo = homophily(knn_edge_index, y)
        print(kernel_name[i], k + 1, 'edge_homo:',
              edge_homo, 'node_homo:', node_homo)

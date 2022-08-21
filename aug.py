import numpy as np
import torch
from torch_geometric.utils.dropout import dropout_adj
import time


def remove_edge(edge_index, drop_ratio):
    edge_index, _ = dropout_adj(edge_index, p = drop_ratio)

    return edge_index


def drop_node(x, drop_ratio):
    node_num, _ = x.size()
    drop_num = int(node_num * drop_ratio)

    idx_mask = np.random.choice(node_num, drop_num, replace = False).tolist()

    x[idx_mask] = 0

    return x


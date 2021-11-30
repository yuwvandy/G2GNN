import torch
import random
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops, degree, dropout_adj


def upsample(H, y, num_training_graph):
    max_num_training_graph = max(num_training_graph)
    classes = torch.unique(y)

    chosen = []
    for i in range(len(classes)):
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()

        up_sample_ratio = max_num_training_graph / num_training_graph[i]
        up_sample_num = int(
            num_training_graph[i] * up_sample_ratio - num_training_graph[i])

        if(up_sample_num <= len(train_idx)):
            up_sample = random.sample(train_idx, up_sample_num)
        else:
            tmp = int(up_sample_num / len(train_idx))
            up_sample = train_idx * tmp
            tmp = up_sample_num - len(train_idx) * tmp

            up_sample.extend(random.sample(train_idx, tmp))

        chosen.extend(up_sample)

    chosen = torch.tensor(chosen)
    H = torch.cat([H, H[chosen]], dim = 0)
    y = torch.cat([y, y[chosen]], dim = 0)

    return H, y


def embed_smote(embed, num_training_graph, y, k):
    max_num_training_graph = max(num_training_graph)
    classes = torch.unique(y)

    embed_aug = []
    y_aug = []

    for i in range(len(classes)):
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()

        c_embed = embed[train_idx]
        c_dist = torch.cdist(c_embed, c_embed, p=2)

        # different from original smote, we select furtherst majority nodes
        c_min_idx = torch.topk(c_dist, min(k, c_dist.size(0)), largest=False)[
            1][:, 1:].tolist()

        up_sample_ratio = max_num_training_graph / \
            num_training_graph[i]
        up_sample_num = int(
            num_training_graph[i] * up_sample_ratio - num_training_graph[i])


        tmp = 1
        head_list = list(np.arange(0, len(train_idx)))

        while(tmp <= up_sample_num):
            head_id = random.choice(head_list)
            tail_id = random.choice(c_min_idx[head_id])

            delta = torch.rand(1).to(c_embed.device)
            new_embed = torch.lerp(
                c_embed[head_id], c_embed[tail_id], delta)

            embed_aug.append(new_embed)
            y_aug.append(classes[i])

            tmp += 1

    return torch.stack(embed_aug), torch.stack(y_aug).to(embed.device)


def mixup(x, y, args, alpha=1.0):
    """
    modified from the original code for that mixup paper
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def remixup(x, y, num_training_graph, alpha=1.0, tau=3, kappa=3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    num_class_a, num_class_b = num_training_graph[y_a], num_training_graph[y_b]
    num_ratio = num_class_a / num_class_b
    cond_ratio1, cond_ratio2 = num_ratio >= kappa, num_ratio <= 1 / kappa

    lamb_list = mixed_x.new(batch_size).fill_(lam)
    cond_lamb1 = lamb_list < tau
    cond_lamb2 = (1 - lamb_list) < tau

    lamb_list[cond_lamb1 & cond_ratio1] = 0
    lamb_list[cond_lamb2 & cond_ratio2] = 1

    return mixed_x, y_a, y_b, lamb_list



def construct_knn(kernel_idx):
    edge_index = [[], []]

    for i in range(len(kernel_idx)):
        for j in range(len(kernel_idx[i])):
            edge_index[0].append(i)
            edge_index[1].append(kernel_idx[i, j].item())

            edge_index[0].append(kernel_idx[i, j].item())
            edge_index[1].append(i)

    return torch.tensor(edge_index, dtype = torch.long)

def propagate(edge_index, x):
    """ feature propagation procedure: sparsematrix
    """

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[row]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')

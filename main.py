from model import *
from learn import *
from dataset import *

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_remaining_self_loops, degree, to_networkx, to_scipy_sparse_matrix
import networkx as nx

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath


import numpy as np
import argparse
from tqdm import tqdm
import random
import math
from os import path


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--runs', type=int, default=50)
parser.add_argument('--imb_ratio', type=float, default=0.1)


parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_training', type=int, default=100)
parser.add_argument('--num_val', type=int, default=30)

parser.add_argument('--setting', type=str, default='aug')
parser.add_argument('--aug', type=str, default='RE')
parser.add_argument('--aug_ratio', type=float, default=0.1)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--prop_epochs', type=int, default=3)
parser.add_argument('--knn', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--kernel_type', type=str, default='SP')
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--aug_num', type=int, default=2)
parser.add_argument('--temp', type=float, default=0.5)

args = parser.parse_args()

dataset, args.n_feat, args.n_class, args.mapping = get_TUDataset(args.dataset)

# ====== compute the kernel and select the top k
if(args.setting in ['knn', 'knn_aug']):
    kernel_file = 'kernel/' + args.dataset + args.kernel_type + '.txt'
    if(path.exists(kernel_file)):
        args.kernel_simi = torch.load(kernel_file)
    else:
        dataset2 = fetch_dataset(args.dataset, verbose=False)
        G = dataset2.data
        if(args.dataset in ['IMDB-BINARY', 'REDDIT-BINARY']):
            gk = ShortestPath(normalize=True, with_labels=False)
        else:
            gk = ShortestPath(normalize=True)
        args.kernel_simi = torch.tensor(gk.fit_transform(G))
        torch.save(args.kernel_simi, kernel_file)

    args.kernel_idx = torch.topk(
        args.kernel_simi, k=args.knn, dim=1, largest=True)[1][:, 1:]
    args.knn_edge_index = construct_knn(args.kernel_idx)


F1_micro = np.zeros(args.runs, dtype=float)
F1_macro = np.zeros(args.runs, dtype=float)

pbar = tqdm(range(args.runs), unit='run')

for count in pbar:
    random.seed(args.seed + count)
    np.random.seed(args.seed + count)
    torch.manual_seed(args.seed + count)
    torch.cuda.manual_seed_all(args.seed + count)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset, val_dataset, test_dataset, args.class_train_num_graph, args.class_val_num_graph = shuffle(
        args.dataset, dataset, args.imb_ratio, args.num_training, args.num_val)

    args.min_num_graph = min(args.class_train_num_graph)

    if(args.setting in ['upsampling', 'knn', 'knn_aug', 'aug'] and args.min_num_graph != 0 and args.imb_ratio != 0.5):
        train_dataset = upsample(train_dataset)
        val_dataset = upsample(val_dataset)

    # if(args.setting == 'reweight'): #if want to pre-reweighting
    #     args.weight = max(args.class_train_num_graph) / \
    #         args.class_train_num_graph

    train_label = torch.tensor([data.y.item() for data in train_dataset])
    val_label = torch.tensor([data.y.item() for data in val_dataset])
    test_label = torch.tensor([data.y.item() for data in test_dataset])

    # print([(train_label == i).sum().item() for i in range(args.n_class)])
    # print([(val_label == i).sum().item() for i in range(args.n_class)])
    # print([(test_label == i).sum().item() for i in range(args.n_class)])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = GIN(args).to(args.device)
    classifier = MLP_Classifier(args).to(args.device)
    args.class_train_num_graph = args.class_train_num_graph.to(args.device)
    if(args.setting in ['knn', 'knn_aug']):
        args.knn_edge_index = args.knn_edge_index.to(args.device)
    # if(args.setting == 'reweight'):
    #     args.weight = args.weight.to(args.device)

    optimizer_e = torch.optim.Adam(
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_c = torch.optim.Adam(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = math.inf
    val_loss_hist = []
    for epoch in range(1, args.epochs):
        loss = train(epoch, encoder, classifier, train_loader, dataset,
                     optimizer_e, optimizer_c, args)

        val_eval = eval(encoder, classifier, val_loader, dataset,
                        optimizer_e, optimizer_c, args)

        if(val_eval['loss'] < best_val_loss):
            best_val_loss = val_eval['loss']

            test_eval = eval(encoder, classifier, test_loader, dataset,
                             optimizer_e, optimizer_c, args)

            #torch.save(encoder.state_dict(), 'encoder' + args.setting + '.pkl')
            # torch.save(classifier.state_dict(),
            #'classifier' + args.setting + '.pkl')

        # print(test_eval)

        val_loss_hist.append(val_eval['loss'])

        if(args.early_stopping > 0 and epoch > args.epochs // 2):
            tmp = torch.tensor(val_loss_hist[-(args.early_stopping + 1): -1])
            if(val_eval['loss'] > tmp.mean().item()):
                break

    F1_micro[count] = test_eval['F1-micro']
    F1_macro[count] = test_eval['F1-macro']

    # print('F1_micro:', np.mean(
    #     F1_micro[:(count + 1)]), 'std:', np.std(F1_micro[:(count + 1)]), 'F1-macro:', np.mean(F1_macro[:(count + 1)]), 'std:', np.std(F1_macro[:(count + 1)]))


print('F1_macro: ', np.mean(F1_macro), np.std(F1_macro))
print('F1_micro: ', np.mean(F1_micro), np.std(F1_micro))

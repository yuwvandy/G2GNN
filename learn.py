from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch_geometric.utils import subgraph, degree

from utils import *
from aug import *

import time


def train(epoch, encoder, classifier, data_loader, dataset, optimizer_e, optimizer_c, args):
    encoder.train()
    classifier.train()

    total_loss = 0
    for data in data_loader:
        if(args.setting in ['no', 'upsampling'] or args.min_num_graph == 0):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)

            loss = F.nll_loss(logits, data.y)

        elif(args.setting == 'reweight'):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)

            args.weight = torch.zeros_like(
                args.class_train_num_graph, dtype=torch.float)
            for i in range(len(torch.unique(data.y))):
                args.weight[i] = 1 / \
                    ((data.y == torch.unique(data.y)[i]).sum())

            loss = F.nll_loss(logits, data.y, weight=args.weight)

        elif(args.setting == 'smote'):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)

            H_aug, y_aug = embed_smote(
                H, args.class_train_num_graph, data.y, args.k)
            logits_aug = classifier(H_aug)

            loss = F.nll_loss(logits, data.y) + \
                F.nll_loss(logits_aug, y_aug)

        elif(args.setting == 'mix'):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)

            mixed_H, y_a, y_b, lam = mixup(H, data.y, args, alpha=1.0)
            logits_mix = classifier(mixed_H)

            loss = lam * F.nll_loss(logits_mix, y_a) + \
                (1 - lam) * F.nll_loss(logits_mix, y_b)

        elif(args.setting == 'remix'):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            mixed_H, y_a, y_b, lam = remixup(
                H, data.y, args.class_train_num_graph, alpha=1.0, tau=0.5, kappa=3)
            logits_mix = classifier(mixed_H)

            loss = F.nll_loss(lam.view(-1, 1) * logits_mix, y_a) + \
                F.nll_loss((1 - lam).view(-1, 1) * logits_mix, y_b)

        elif(args.setting == 'knn'):
            idx = len(data.id)
            extra_id = id_pad(data.id, args.kernel_idx)

            if(len(extra_id) != 0):
                data = data_pad(data, extra_id, dataset,
                                args.mapping).to(args.device)
            else:
                data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)[:idx]

            knn_edge_index, _ = subgraph(
                data.id, args.knn_edge_index, relabel_nodes=True)

            y_data = data.y[:idx]

            H_knn = H

            for i in range(args.prop_epochs):
                H_knn = propagate(knn_edge_index, H_knn)

            logits_knn = classifier(H_knn)[:idx]
            loss = F.nll_loss(logits_knn, y_data)

        elif(args.setting == 'aug'):
            data = data.to(args.device)

            H_augs, logit_augs = [], []
            for i in range(args.aug_num):
                if(args.aug == 'RE'):
                    edge_index_aug = remove_edge(
                        data.edge_index, args.aug_ratio)
                    H_augs.append(encoder(data.x, edge_index_aug, data.batch))
                    logit_augs.append(classifier(H_augs[-1]))
                elif(args.aug == 'DN'):
                    x_aug = drop_node(data.x, args.aug_ratio)
                    H_augs.append(encoder(x_aug, data.edge_index, data.batch))
                    logit_augs.append(classifier(H_augs[-1]))

            loss = 0
            for i in range(len(logit_augs)):
                loss += F.nll_loss(logit_augs[i], data.y)
            loss = loss / len(logit_augs)

            loss = loss + consis_loss(logit_augs, temp=args.temp)

        elif(args.setting == 'knn_aug'):
            idx = len(data.id)
            extra_id = id_pad(data.id, args.kernel_idx)

            if(len(extra_id) != 0):
                data = data_pad(data, extra_id, dataset,
                                args.mapping).to(args.device)
            else:
                data = data.to(args.device)

            H_augs, logit_augs = [], []
            for i in range(args.aug_num):
                if(args.aug == 'RE'):
                    edge_index_aug = remove_edge(
                        data.edge_index, args.aug_ratio)
                    H_augs.append(encoder(data.x, edge_index_aug, data.batch))
                elif(args.aug == 'DN'):
                    x_aug = drop_node(data.x, args.aug_ratio)
                    H_augs.append(encoder(x_aug, data.edge_index, data.batch))

            knn_edge_index, _ = subgraph(
                data.id, args.knn_edge_index, relabel_nodes=True)

            logit_aug_props = []
            for k in range(args.aug_num):
                H_aug_knn = H_augs[k]
                for i in range(args.prop_epochs):
                    H_aug_knn = propagate(knn_edge_index, H_aug_knn)
                logit_aug_props.append(classifier(H_aug_knn[:idx]))

            y_data = data.y[:idx]

            loss = 0
            for i in range(len(logit_aug_props)):
                loss += F.nll_loss(logit_aug_props[i], y_data)
            loss = loss / len(logit_aug_props)

            loss = loss + consis_loss(logit_aug_props)

        optimizer_e.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer_e.step()
        optimizer_c.step()

        total_loss += float(loss) * data.num_graphs

    return total_loss / len(data_loader.dataset)


@ torch.no_grad()
def eval(encoder, classifier, data_loader, dataset, optimizer_e, optimizer_c, args):
    encoder.eval()
    classifier.eval()

    pred, truth = [], []
    total_loss = 0
    for data in data_loader:
        if(args.setting in ['no', 'upsampling'] or args.min_num_graph == 0):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)

            loss = F.nll_loss(logits, data.y)

            pred.extend(logits.argmax(-1).tolist())
            truth.extend(data.y.tolist())

        elif(args.setting == 'reweight'):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)

            args.weight = max(args.class_train_num_graph) / \
                args.class_train_num_graph
            loss = F.nll_loss(logits, data.y, weight=args.weight)

            pred.extend(logits.argmax(-1).tolist())
            truth.extend(data.y.tolist())

        elif(args.setting == 'smote'):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)

            H_aug, y_aug = embed_smote(
                H, args.class_train_num_graph, data.y, args.k)
            logits_aug = classifier(H_aug)

            loss = F.nll_loss(logits, data.y) + \
                F.nll_loss(logits_aug, y_aug)

            pred.extend(logits.argmax(-1).tolist())
            truth.extend(data.y.tolist())

        elif(args.setting == 'mix'):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)

            mixed_H, y_a, y_b, lam = mixup(H, data.y, args, alpha=1.0)
            logits_mix = classifier(mixed_H)

            loss = lam * F.nll_loss(logits_mix, y_a) + \
                (1 - lam) * F.nll_loss(logits_mix, y_b)

            pred.extend(logits.argmax(-1).tolist())
            truth.extend(data.y.tolist())

        elif(args.setting == 'remix'):
            data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            mixed_H, y_a, y_b, lam = remixup(
                H, data.y, args.class_train_num_graph, alpha=1.0, tau=0.5, kappa=3)
            logits_mix = classifier(mixed_H)

            loss = F.nll_loss(lam.view(-1, 1) * logits_mix, y_a) + \
                F.nll_loss((1 - lam).view(-1, 1) * logits_mix, y_b)

            pred.extend(logits_mix.argmax(-1).tolist())
            truth.extend(data.y.tolist())

        elif(args.setting == 'knn'):
            idx = len(data.id)
            extra_id = id_pad(data.id, args.kernel_idx)
            if(len(extra_id) != 0):
                data = data_pad(data, extra_id, dataset,
                                args.mapping).to(args.device)
            else:
                data = data.to(args.device)

            H = encoder(data.x, data.edge_index, data.batch)
            logits = classifier(H)[:idx]

            knn_edge_index, _ = subgraph(
                data.id, args.knn_edge_index, relabel_nodes=True)

            y_data = data.y[:idx]

            H_knn = H
            for i in range(args.prop_epochs):
                H_knn = propagate(knn_edge_index, H_knn)

            logits_knn = classifier(H_knn)[:idx]
            loss = F.nll_loss(logits_knn, y_data)

            pred.extend(logits_knn.argmax(-1).tolist())
            truth.extend(y_data.tolist())

        elif(args.setting == 'aug'):
            data = data.to(args.device)

            H_augs, logit_augs = [], []
            for i in range(args.aug_num):
                if(args.aug == 'RE'):
                    edge_index_aug = remove_edge(
                        data.edge_index, args.aug_ratio)
                    H_augs.append(encoder(data.x, edge_index_aug, data.batch))
                    logit_augs.append(classifier(H_augs[-1]))
                elif(args.aug == 'DN'):
                    x_aug = drop_node(data.x, args.aug_ratio)
                    H_augs.append(encoder(x_aug, data.edge_index, data.batch))
                    logit_augs.append(classifier(H_augs[-1]))

            loss = 0
            for i in range(len(logit_augs)):
                loss += F.nll_loss(logit_augs[i], data.y)
            loss = loss / len(logit_augs)
            loss = loss + consis_loss(logit_augs)

            logit_aug = torch.stack(logit_augs).mean(dim=0)

            pred.extend(logit_aug.argmax(-1).tolist())
            truth.extend(data.y.tolist())

        elif(args.setting == 'knn_aug'):
            idx = len(data.id)
            extra_id = id_pad(data.id, args.kernel_idx)

            if(len(extra_id) != 0):
                data = data_pad(data, extra_id, dataset,
                                args.mapping).to(args.device)
            else:
                data = data.to(args.device)

            H_augs, logit_augs = [], []
            for i in range(args.aug_num):
                if(args.aug == 'RE'):
                    edge_index_aug = remove_edge(
                        data.edge_index, args.aug_ratio)
                    H_augs.append(encoder(data.x, edge_index_aug, data.batch))
                elif(args.aug == 'DN'):
                    x_aug = drop_node(data.x, args.aug_ratio)
                    H_augs.append(encoder(x_aug, data.edge_index, data.batch))

            knn_edge_index, _ = subgraph(
                data.id, args.knn_edge_index, relabel_nodes=True)

            logit_aug_props = []
            for k in range(args.aug_num):
                H_aug_knn = H_augs[k]
                for i in range(args.prop_epochs):
                    H_aug_knn = propagate(knn_edge_index, H_aug_knn)
                logit_aug_props.append(classifier(H_aug_knn[:idx]))

            y_data = data.y[:idx]

            loss = 0
            for i in range(len(logit_aug_props)):
                loss += F.nll_loss(logit_aug_props[i], y_data)
            loss = loss / len(logit_aug_props)

            loss = loss + consis_loss(logit_aug_props)

            logit_aug_prop = torch.stack(logit_aug_props).mean(dim=0)

            pred.extend(logit_aug_prop.argmax(-1).tolist())
            truth.extend(y_data.tolist())

        total_loss += float(loss) * data.num_graphs

    acc_c = f1_score(truth, pred, labels=np.arange(
        0, 2), average=None, zero_division=0)
    acc = (np.array(pred) == np.array(truth)).sum() / len(truth)

    return {'loss': total_loss / len(data_loader.dataset), 'F1-macro': np.mean(acc_c), 'F1-micro': acc}

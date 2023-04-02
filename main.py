import os
from parse import parse_args
from tqdm import tqdm
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import math

from utils import *
from model import *
from learn import *
from dataset import *
from dataprocess import *


def run(args):
    pbar = tqdm(range(args.runs), unit='run')

    F1_micro = np.zeros(args.runs, dtype=float)
    F1_macro = np.zeros(args.runs, dtype=float)

    if args.setting in ['upsampling', 'no', 'overall_reweight', 'batch_reweight', 'smote']:
        Dataset_tmp = Dataset
    elif args.setting == 'knn':
        Dataset_tmp = Dataset_knn
    elif args.setting == 'aug':
        Dataset_tmp = Dataset_aug
    elif args.setting == 'knn_aug':
        Dataset_tmp = Dataset_knn_aug

    for count in pbar:
        random.seed(args.seed + count)
        np.random.seed(args.seed + count)
        torch.manual_seed(args.seed + count)
        torch.cuda.manual_seed_all(args.seed + count)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_data, val_data, test_data = shuffle(
            dataset, args.c_train_num, args.c_val_num, args.y)

        if args.setting in ['upsampling', 'knn', 'knn_aug', 'aug']:
            train_data = upsample(train_data)
            val_data = upsample(val_data)

        train_dataset = Dataset_tmp(train_data, dataset, args)
        val_dataset = Dataset_tmp(val_data, dataset, args)
        test_dataset = Dataset_tmp(test_data, dataset, args)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_dataset.collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=val_dataset.collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=test_dataset.collate_batch)

        encoder = GIN(args).to(args.device)
        classifier = MLP_Classifier(args).to(args.device)

        optimizer_e = torch.optim.Adam(
            encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_c = torch.optim.Adam(
            classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_loss = math.inf
        val_loss_hist = []

        for epoch in range(0, args.epochs):
            loss = train(encoder, classifier, train_loader,
                         optimizer_e, optimizer_c, args)
            val_eval = eval(encoder, classifier, val_loader, args)

            if(val_eval['loss'] < best_val_loss):
                best_val_loss = val_eval['loss']

                test_eval = eval(encoder, classifier, test_loader, args)

            val_loss_hist.append(val_eval['loss'])

            if(args.early_stopping > 0 and epoch > args.epochs // 2):
                tmp = torch.tensor(
                    val_loss_hist[-(args.early_stopping + 1): -1])
                if(val_eval['loss'] > tmp.mean().item()):
                    break

        F1_micro[count] = test_eval['F1-micro']
        F1_macro[count] = test_eval['F1-macro']

        # print('F1_micro:', np.mean(F1_micro[:(count + 1)]), 'std:', np.std(F1_micro[:(count + 1)]),
        #       'F1-macro:', np.mean(F1_macro[:(count + 1)]), 'std:', np.std(F1_macro[:(count + 1)]))

    return F1_micro, F1_macro


if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset, args.n_feat, args.n_class, _ = get_TUDataset(
        args.dataset, pre_transform=T.ToSparseTensor())

    args.c_train_num, args.c_val_num = get_class_num(
        args.imb_ratio, args.num_train, args.num_val)

    args.y = torch.tensor([data.y.item() for data in dataset])

    if args.setting in ['knn', 'knn_aug']:
        args.kernel_idx, args.knn_edge_index = get_kernel_knn(args.dataset, args.kernel_type, args.knn_nei_num)

    if args.setting in ['overall_reweight']:
        args.weight = torch.zeros(len(args.c_train_num), dtype=torch.float)
        args.weight = sum(args.c_train_num)/torch.tensor(args.c_train_num)


    F1_micro, F1_macro = run(args)

    print('F1_macro: ', np.mean(F1_macro), np.std(F1_macro))
    print('F1_micro: ', np.mean(F1_micro), np.std(F1_micro))

from torch.utils.data import Dataset as BaseDataset
from torch_geometric.data.collate import collate
import torch
from utils import *
from torch_geometric.utils import subgraph, degree
from aug import *
from torch_sparse import SparseTensor, matmul


class Dataset(BaseDataset):
    def __init__(self, dataset, all_dataset, args):
        self.args = args
        self.dataset = dataset
        self.all_dataset = all_dataset

    def _get_feed_dict(self, index):
        feed_dict = self.dataset[index]

        return feed_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        batch_id = torch.tensor([feed_dict.id for feed_dict in feed_dicts])
        # prevent testing data leakage
        train_idx = torch.arange(batch_id.shape[0])

        data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True,
        )

        batch = {'data': data,
                 'train_idx': train_idx}

        return batch




class Dataset_knn(BaseDataset):
    def __init__(self, dataset, all_dataset, args):
        self.args = args
        self.dataset = dataset
        self.all_dataset = all_dataset

    def _get_feed_dict(self, index):
        feed_dict = self.dataset[index]

        return feed_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        batch_id = torch.tensor([feed_dict.id for feed_dict in feed_dicts])
        # prevent testing data leakage
        train_idx = torch.arange(batch_id.shape[0])

        # add_knn_dataset to feed_dicts
        pad_knn_id = find_knn_id(batch_id, self.args.kernel_idx)
        feed_dicts.extend([self.all_dataset[i] for i in pad_knn_id])

        data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True,
        )

        # print(data.id, self.args.knn_edge_index)
        knn_edge_index, _ = subgraph(
            data.id, self.args.knn_edge_index, relabel_nodes=True)

        knn_edge_index, _ = add_remaining_self_loops(knn_edge_index)
        row, col = knn_edge_index
        knn_deg = degree(col, data.id.shape[0])
        deg_inv_sqrt = knn_deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[col]

        knn_adj_t = torch.sparse.FloatTensor(
            knn_edge_index, edge_weight, (data.id.size(0), data.id.size(0)))

        batch = {'data': data,
                 'train_idx': train_idx,
                 'knn_adj_t': knn_adj_t}

        return batch


class Dataset_aug(BaseDataset):
    def __init__(self, dataset, all_dataset, args):
        self.args = args
        self.dataset = dataset
        self.all_dataset = all_dataset

    def _get_feed_dict(self, index):
        feed_dict = self.dataset[index]

        return feed_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        batch_id = torch.tensor([feed_dict.id for feed_dict in feed_dicts])
        # prevent testing data leakage
        train_idx = torch.arange(batch_id.shape[0])

        data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True,
        )

        aug_xs, aug_adj_ts = [], []
        for i in range(self.args.aug_num):
            edge_index = torch.stack(data.adj_t.coo()[:2])
            edge_index_aug = remove_edge(edge_index, self.args.drop_edge_ratio)
            aug_adj_ts.append(SparseTensor(
                row=edge_index_aug[0], col=edge_index_aug[1], value=None, sparse_sizes=(data.x.size(0), data.x.size(0))))

            aug_xs.append(drop_node(data.x, self.args.mask_node_ratio))

        batch = {'data': data,
                 'train_idx': train_idx,
                 'aug_adj_ts': aug_adj_ts,
                 'aug_xs': aug_xs}

        return batch


class Dataset_knn_aug(BaseDataset):
    def __init__(self, dataset, all_dataset, args):
        self.args = args
        self.dataset = dataset
        self.all_dataset = all_dataset

    def _get_feed_dict(self, index):
        feed_dict = self.dataset[index]

        return feed_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        batch_id = torch.tensor([feed_dict.id for feed_dict in feed_dicts])
        # prevent testing data leakage
        train_idx = torch.arange(batch_id.shape[0])

        # add_knn_dataset to feed_dicts
        pad_knn_id = find_knn_id(batch_id, self.args.kernel_idx)
        feed_dicts.extend([self.all_dataset[i] for i in pad_knn_id])

        data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True,
        )

        knn_edge_index, _ = subgraph(
            data.id, self.args.knn_edge_index, relabel_nodes=True)

        knn_edge_index, _ = add_remaining_self_loops(knn_edge_index)
        row, col = knn_edge_index
        knn_deg = degree(col, data.id.shape[0])
        deg_inv_sqrt = knn_deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[col]

        knn_adj_t = torch.sparse.FloatTensor(
            knn_edge_index, edge_weight, (data.id.size(0), data.id.size(0)))


        aug_xs, aug_adj_ts = [], []
        for i in range(self.args.aug_num):
            edge_index = torch.stack(data.adj_t.coo()[:2])
            edge_index_aug = remove_edge(edge_index, self.args.drop_edge_ratio)
            aug_adj_ts.append(SparseTensor(
                row=edge_index_aug[0], col=edge_index_aug[1], value=None, sparse_sizes=(data.x.size(0), data.x.size(0))))

            aug_xs.append(drop_node(data.x, self.args.mask_node_ratio))

        batch = {'data': data,
                 'train_idx': train_idx,
                 'aug_adj_ts': aug_adj_ts,
                 'aug_xs': aug_xs,
                 'knn_adj_t': knn_adj_t}

        return batch

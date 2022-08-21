import torch
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch.nn import Linear
import torch.nn.functional as F
from torch_scatter import segment_csr


class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            Sequential(Linear(args.n_feat, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))


    def forward(self, x, adj_t, batch):
        x = self.conv1(x, adj_t)
        x = self.conv2(x, adj_t)
        x = self.conv3(x, adj_t)
        x = self.conv4(x, adj_t)
        x = self.conv5(x, adj_t)

        x = segment_csr(x, batch, reduce="sum")

        return x


class MLP_Classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_Classifier, self).__init__()
        self.lin1 = Linear(args.n_hidden, args.n_hidden)
        self.lin2 = Linear(args.n_hidden, args.n_class)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)

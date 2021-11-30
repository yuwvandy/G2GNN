import torch
import random
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops, degree, dropout_adj
from torch_geometric.loader import DataLoader
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.rcParams["font.family"] = "Times New Roman"


def upsample(dataset):
    y = torch.tensor([dataset[i].y for i in range(len(dataset))])
    classes = torch.unique(y)

    num_class_graph = [(y == i.item()).sum() for i in classes]

    max_num_class_graph = max(num_class_graph)

    chosen = []
    for i in range(len(classes)):
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()

        up_sample_ratio = max_num_class_graph / num_class_graph[i]
        up_sample_num = int(
            num_class_graph[i] * up_sample_ratio - num_class_graph[i])

        if(up_sample_num <= len(train_idx)):
            up_sample = random.sample(train_idx, up_sample_num)
        else:
            tmp = int(up_sample_num / len(train_idx))
            up_sample = train_idx * tmp
            tmp = up_sample_num - len(train_idx) * tmp

            up_sample.extend(random.sample(train_idx, tmp))

        chosen.extend(up_sample)

    chosen = torch.tensor(chosen)
    extend_data = dataset[chosen]

    data = list(dataset) + list(extend_data)

    return data


def embed_smote(embed, num_training_graph, y, k):
    max_num_training_graph = max(num_training_graph)
    classes = torch.unique(y)

    embed_aug = []
    y_aug = []

    for i in range(len(classes)):
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()

        c_embed = embed[train_idx]
        c_dist = torch.cdist(c_embed, c_embed, p=2)

        # different from original smote, we also include itself in case of no other nodes to use
        c_min_idx = torch.topk(c_dist, min(k, c_dist.size(0)), largest=False)[
            1][:, :].tolist()

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

    if(embed_aug == []):
        return embed, y

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

    return torch.tensor(edge_index, dtype=torch.long)


def propagate(edge_index, x, deg=None):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index)
    # calculate the degree normalize term
    if(deg == None):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
        edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[col]
    # print(edge_weight)
    else:
        row, col = edge_index
        deg_inv_sqrt = deg.pow(-0.5).to(edge_index.device)
        edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def id_pad(data_id, kernel_id):
    pad_id = set(kernel_id[data_id].view(-1).tolist())
    repeat_id = pad_id.intersection(set(data_id.tolist()))

    extra_id = pad_id - repeat_id

    return torch.tensor(list(extra_id), dtype=torch.long)


def data_pad(data, extra_id, dataset, mapping):
    extra_id = [mapping[extra_id[i].item()] for i in range(len(extra_id))]
    extra_dataset = dataset[extra_id]
    extra_dataloader = DataLoader(extra_dataset, batch_size=len(extra_dataset))

    for extra_data in extra_dataloader:
        if(extra_data.x is None):
            extra_data.x = torch.ones((extra_data.batch.size(0), 1))
        data.edge_index = torch.cat(
            [data.edge_index, extra_data.edge_index + data.x.size(0)], dim=1)
        data.x = torch.cat([data.x, extra_data.x], dim=0)

        data.id = torch.cat([data.id, extra_data.id], dim=0)
        data.y = torch.cat([data.y, extra_data.y], dim=0)
        data.batch = torch.cat(
            [data.batch, extra_data.batch + max(data.batch) + 1])

    return data


def consis_loss(logps, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) /
               torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return 1 * loss


def homophily(edge_index, y):
    degree_cal = degree(edge_index[1], num_nodes=y.size(0))

    edge_homo = (y[edge_index[0]] == y[edge_index[1]]
                 ).sum().item() / edge_index.size(1)

    tmp = y[edge_index[0]] == y[edge_index[1]]
    node_homo = scatter(tmp, edge_index[1], dim=0, dim_size=y.size(
        0), reduce='add') / degree_cal

    return edge_homo, node_homo.mean()


def tsne_visual(encoder, classifier, dataset, args):
    x, y, pred = [], [], []

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size)

    mapping = {}
    inverse_mapping = {}
    count = 0
    for data in data_loader:
        data = data.to(args.device)

        H = encoder(data.x, data.edge_index, data.batch)

        x.append(H)
        y.append(data.y)
        pred.append(classifier(H).argmax(-1))

        for idd in data.id:
            mapping[idd.item()] = count
            inverse_mapping[count] = idd.item()
            count += 1

    x = torch.cat(x, dim=0).detach().cpu().numpy()
    y = torch.cat(y, dim=0).detach().cpu().numpy()
    pred = torch.cat(pred, dim=0).detach().cpu().numpy()
    pred_mask = (pred == y)

    print(x.shape)
    x = TSNE(n_components=2, learning_rate='auto',
             init='random', random_state=0).fit_transform(x)

    plt.subplot(1, 1, 1)
    print(y.shape, pred.shape, (y == 0) & (pred == 1))
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='grey', s=5)
    plt.scatter(x[(y == 0) & (pred == 0), 0],
                x[(y == 0) & (pred == 0), 1], c='red', s=5)
    plt.scatter(x[(y == 0) & (pred == 1), 0],
                x[(y == 0) & (pred == 1), 1], c='red', s=30)

    for edge in args.knn_edge_index.t():
        if(y[mapping[edge[0].item()]] == 0 or y[mapping[edge[1].item()]] == 0):  # min nodes
            if(pred_mask[mapping[edge[0].item()]] == False or pred_mask[mapping[edge[1].item()]] == False):
                if(y[mapping[edge[0].item()]] == 0 and y[mapping[edge[1].item()]] == 0):
                    c = 'red'
                if(y[mapping[edge[0].item()]] == 1 and y[mapping[edge[1].item()]] == 1):
                    c = 'grey'
                if(y[mapping[edge[0].item()]] != y[mapping[edge[1].item()]]):
                    c = 'orange'
                plt.plot((x[mapping[edge[0].item()], 0], x[mapping[edge[1].item()], 0]), (
                    x[mapping[edge[0].item()], 1], x[mapping[edge[1].item()], 1]), linewidth=0.2, c=c)
    # plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], c = 'red', s = 30, edgecolor = 'k')
    # plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], c = 'grey', s = 30, edgecolor = 'k')
    # plt.xlabel('1st Principal Component', fontweight = 'bold', fontsize = 20)
    # plt.ylabel('2nd Principal Component', fontweight = 'bold', fontsize = 20)
    # plt.title('Scatter Plot with Decision Boundary for the Test Set')
    plt.savefig('./result/visual' + args.setting + '.pdf', dpi=1000)


def pca_visual(encoder, classifier, dataset, args):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import IncrementalPCA
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import seaborn as sns
    # %matplotlib inline

    x = []
    y = []

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size)

    y_pred = []

    mapping = {}
    inverse_mapping = {}
    count = 0
    for data in data_loader:
        data = data.to(args.device)
        H = encoder(data.x, data.edge_index, data.batch)
        x.append(H)
        y.append(data.y)
        y_pred.append(classifier(H).argmax(-1))

        for idd in data.id:
            mapping[idd.item()] = count
            inverse_mapping[count] = idd.item()
            count += 1

    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    x = torch.cat(x, dim=0).detach().cpu().numpy()
    y = torch.cat(y, dim=0).detach().cpu().numpy()

    print(len(y), (y_pred == y).sum() / len(y))

    scalerx = StandardScaler()
    X_scaled = scalerx.fit_transform(x)

    pca = IncrementalPCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # plt.figure(figsize = (20, 6))
    # plt.subplot(1, 2, 1)
    # plt.scatter(X_pca[:,0], X_pca[:,1], c = y_train)
    # plt.xlabel('Training 1st Principal Component')
    # plt.ylabel('Training 2nd Principal Component')
    # plt.title('Training Set Scatter Plot with labels indicated by colors i.e., (0) -> Violet and (1) -> Yellow')
    # plt.subplot(1, 2, 2)
    # plt.scatter(X_test_pca[:,0], X_test_pca[:,1], c = y_test)
    # plt.xlabel('Test 1st Principal Component')
    # plt.ylabel('Test 2nd Principal Component')
    # plt.title('Test Set Scatter Plot with labels indicated by colors i.e., (0) -> Violet and (1) -> Yellow')
    # plt.show()

    params = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}

    clf = LogisticRegression()

    folds = 5
    model = GridSearchCV(estimator=clf,
                         param_grid=params,
                         scoring='accuracy',
                         cv=folds,
                         return_train_score=True,
                         verbose=3)

    model.fit(X_pca, y)

    # model = LogisticRegression(C = 10, random_state=1).fit(X_train_pca, y_train)

    # getting the Training Set Predictions
    # y_train_pred = model.predict(X_train_pca)

    # getting the Test Set Predictions
    # y_test_pred = model.predict(X_test_pca)

    # y_pred = np.concatenate((y_train_pred, y_test_pred))

    # print('Training Accuracy of the Model: ', metrics.accuracy_score(y_train, y_train_pred))
    # print('Test Accuracy of the Model: ', metrics.accuracy_score(y_test, y_test_pred))
    # print()

    # # Getting the Training and Test Precision of the Logistic Regression Model
    # print('Training Precision of the Model: ', metrics.precision_score(y_train, y_train_pred))
    # print('Test Precision of the Model: ', metrics.precision_score(y_test, y_test_pred))
    # print()

    # # Getting the Training and Test Recall of the Logistic Regression Model
    # print('Training Recall of the Model: ', metrics.recall_score(y_train, y_train_pred))
    # print('Test Recall of the Model: ', metrics.recall_score(y_test, y_test_pred))
    # print()

    # # Getting the Training and Test F1-Score of the Logistic Regression Model
    # print('Training F1-Score of the Model: ', metrics.f1_score(y_train, y_train_pred))
    # print('Test F1-Score of the Model: ', metrics.f1_score(y_test, y_test_pred))
    # print()

    pred_mask = (y_pred == y)
    print(111, pred_mask.sum(), len(pred_mask))

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Z_train = model.predict(np.c_[xx_train.ravel(), yy_train.ravel()])
    # Z_train = Z_train.reshape(xx_train.shape)

    # x_min2, x_max2 = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    # y_min2, y_max2 = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1

    # x_min = min(x_min1, x_min2)
    # x_max = max(x_max1, x_max2)
    # y_min = min(y_min1, y_min2)
    # y_max = max(y_max1, y_max2)

    # xx_test, yy_test = np.meshgrid(np.arange(x_min2, x_max2, 0.1),
    #                                np.arange(y_min2, y_max2, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.contourf(xx_train, yy_train, Z_train, cmap='Pastel1')
    # plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], c = 'red', s = 30, edgecolor = 'k')
    # plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], c = 'grey', s = 30, edgecolor = 'k')
    # plt.xlabel('Training 1st Principal Component')
    # plt.ylabel('Training 2nd Principal Component')
    # plt.title('Scatter Plot with Decision Boundary for the Training Set')
    plt.subplot(1, 1, 1)
    plt.contourf(xx, yy, Z, cmap='Pastel1')
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
                c='red', s=30, edgecolor='k')
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
                c='grey', s=30, edgecolor='k')

    for edge in args.knn_edge_index.t():
        if(pred_mask[mapping[edge[0].item()]] == False or pred_mask[mapping[edge[1].item()]] == False):
            if(y[mapping[edge[0].item()]] == 0 and y[mapping[edge[1].item()]] == 0):
                c = 'red'
            if(y[mapping[edge[0].item()]] == 1 and y[mapping[edge[1].item()]] == 1):
                c = 'grey'
            if(y[mapping[edge[0].item()]] != y[mapping[edge[1].item()]]):
                c = 'orange'
            plt.plot((X_pca[mapping[edge[0].item()], 0], X_pca[mapping[edge[1].item()], 0]), (
                X_pca[mapping[edge[0].item()], 1], X_pca[mapping[edge[1].item()], 1]), linewidth=0.2, c=c)
    # plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], c = 'red', s = 30, edgecolor = 'k')
    # plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], c = 'grey', s = 30, edgecolor = 'k')
    plt.xlabel('1st Principal Component', fontweight='bold', fontsize=20)
    plt.ylabel('2nd Principal Component', fontweight='bold', fontsize=20)
    # plt.title('Scatter Plot with Decision Boundary for the Test Set')
    plt.savefig('db_visual' + args.setting + '.pdf', dpi=1000)

    # legend_elements1 = [Patch(facecolor='red', edgecolor='r',
    #                      label='MUTAG'),
    #                Patch(facecolor='blue', edgecolor='blue',
    #                      label='DHFR'),
    #                Patch(facecolor='green', edgecolor='green',
    #                      label='NCI1'),
    #                Patch(facecolor='orange', edgecolor='orange',
    #                      label='REDDIT-BINARY'),
    #                Patch(facecolor='black', edgecolor='black',
    #                      label='Citeseer')]

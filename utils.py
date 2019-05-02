import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from sklearn.model_selection import StratifiedShuffleSplit

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN", gamma=1):
    adj_normalizer = fetch_normalization(normalization)
    if 'Aug' in normalization:
        adj = adj_normalizer(adj, gamma=gamma)
    elif 'Restart' in normalization:
        adj = adj_normalizer(adj, gamma=gamma)
    else:
        adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="FirstOrderGCN", cuda=True, gamma=1):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, normalization, gamma=gamma)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def sgc_precompute(features, adj, degree, concat):
    t = perf_counter()
    mem = [features]
    for i in range(degree):
        features = torch.spmm(adj, features)
        mem.append(features)
    if concat:
        features = torch.cat(mem, dim=1)
    precompute_time = perf_counter()-t
    return features, precompute_time

def rw_restart_precompute(features, adj, degree, alpha):
    t = perf_counter()
    adj = (1-alpha) * adj
    # adj_power = sp.eye(adj.shape[0]).tocoo()
    # adj_power = sparse_mx_to_torch_sparse_tensor(adj_power)
    adj_power = np.eye(adj.shape[0])
    adj_power = torch.tensor(adj_power, dtype=torch.float32)
    # adj_sum = sp.coo_matrix(adj.shape)
    # adj_sum = sparse_mx_to_torch_sparse_tensor(adj_sum)
    adj_sum = torch.zeros(adj.shape, dtype=torch.float32)
    # print('adj_sum: ', type(adj_sum), 'shape: ', adj_sum.shape)
    # print('adj_power: ', type(adj_power), 'shape: ', adj_power.shape)
    # print('adj: ', type(adj), 'shape: ', adj.shape)
    # adj = adj.to_dense()
    # adj_power = adj_power.to_dense()
    for i in range(degree):
        adj_sum = torch.add(adj_sum, alpha*adj_power)
        adj_power = torch.spmm(adj, adj_power) # bug, cannot handle sparse matrix
        # adj_power = torch.matmul(adj, adj_power)
    features = torch.spmm(torch.add(adj_sum,adj_power),features)
    # features = torch.matmul(torch.add(adj_sum,adj_power),features)
    precompute_time = perf_counter()-t
    return features, precompute_time



def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True, gamma=1.0):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    if "Aug" in normalization:
        adj = adj_normalizer(adj, gamma)
    else:
        adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    if "Aug" in normalization:
        train_adj = adj_normalizer(train_adj, gamma)
    else:
        train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index

def exclude_idx(idx, idx_exclude_list):
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])

def train_val_split(visable_idx, visable_y, ntrain_per_class=20, nstopping = 500, seed=1):
    rnd_state = np.random.RandomState(seed)
    labels = np.argmax(visable_y, axis=1)
    train_idx_split = []
    for i in range(max(labels) + 1):
        train_idx_split.append(rnd_state.choice(
                visable_idx[labels == i], ntrain_per_class, replace=False))
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
            exclude_idx(visable_idx, [train_idx]),
            nstopping, replace=False)
    return train_idx, stopping_idx


def test_split(dataset_str="cora", seed = 1):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    total_y = np.vstack((ally, ty))
    total_x = sp.vstack((allx, tx))
    labels = np.argmax(total_y, axis=1)
    skf = StratifiedShuffleSplit(n_splits=10, test_size=1000, random_state=seed)
    idx_splits = []
    for visable_idx, test_idx in skf.split(total_x,labels):
        visable_x, test_x = total_x[visable_idx], total_x[test_idx]
        visable_y, test_y = total_y[visable_idx], total_y[test_idx]
        train_idx, stopping_idx = train_val_split(visable_idx,visable_y)
        idx_splits.append({'train_idx':train_idx, 'val_idx': stopping_idx, 'test_idx':test_idx})

    return idx_splits

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed, rw_restart_precompute, test_split
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

# Arguments
args = get_citation_args()

# setting random seeds
print(args)
set_seed(args.seed, args.cuda)
alpha = 0.05
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda, gamma=args.gamma, alpha=alpha)
# if normalization is not RWalkRestartS, the param alpha is useless in load_citation function.

if args.model == "SGC":
    if args.normalization != 'RWalkRestart':
        features, precompute_time = sgc_precompute(features, adj, args.degree, args.concat)
    else:
        if args.multi_scale:
            alpha_list = [0.05,0.1,0.15,0.2]
            concat_feats = []
            all_pre_time = 0
            for alpha in alpha_list:
                features, precompute_time = rw_restart_precompute(features, adj, args.degree, alpha, args.concat)
                concat_feats.append(features)
                all_pre_time += precompute_time
            features = torch.cat(concat_feats,dim=1)
            precompute_time = all_pre_time
            if args.multiply_degree:
                row_sum = torch.sparse.sum(adj, dim=1).to_dense()
                row_sum = row_sum.reshape(row_sum.shape[0], -1)
                # print(row_sum.shape)
                # maximum = torch.max(row_sum)
                # row_sum = row_sum / maximum
                # print(features.shape)
                features = torch.mul(features,row_sum)
        else:
            alpha = 0.05
            features, precompute_time = rw_restart_precompute(features, adj, args.degree, alpha, args.concat)
            if args.multiply_degree:
                row_sum = torch.sparse.sum(adj, dim=1).to_dense()
                row_sum = row_sum.reshape(row_sum.shape[0], -1)
                # print(row_sum.shape)
                # maximum = torch.max(row_sum)
                # row_sum = row_sum / maximum
                # print(features.shape)
                features = torch.mul(features,row_sum)
    print("{:.4f}s".format(precompute_time))

model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if args.model == 'SGC':
            output = model(train_features)
        # if args.model == 'GCN':
        #     output = model(adj, train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)

if args.model == "SGC":
    k_fold = True
    acc_test_list = []
    acc_val_list = []
    train_time_list = []
    if k_fold == True:
        idx_splits = test_split(args.dataset)
        idx_splits.append({'train_idx':idx_train,'val_idx':idx_val, 'test_idx':idx_test})
        for idxs in idx_splits:
            idx_train, idx_val, idx_test = idxs['train_idx'], idxs['val_idx'], idxs['test_idx']
            model = get_model(args.model, features.size(1), labels.max().item() + 1, args.hidden, args.dropout,
                              args.cuda)
            model, cur_acc_val, cur_train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                                                          args.epochs, args.weight_decay, args.lr, args.dropout)
            cur_acc_test = test_regression(model, features[idx_test], labels[idx_test])
            acc_test_list.append(cur_acc_test)
            acc_val_list.append(cur_acc_val)
            train_time_list.append(cur_train_time)
        acc_test = np.average(acc_test_list)
        acc_val = np.average(acc_val_list)
        train_time = np.average(train_time_list)
    else:
        model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                                                      args.epochs, args.weight_decay, args.lr, args.dropout)
        acc_test = test_regression(model, features[idx_test], labels[idx_test])

if args.model == "GCN":
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                                                  args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))

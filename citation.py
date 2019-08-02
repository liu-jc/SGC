import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
import os

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
print("{:.4f}s".format(precompute_time))

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    forward_time = 0
    cross_entropy_time = 0
    backward_time = 0
    step_time = 0
    softmax_time = 0
    nll_time = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # forward time
        t_forward = perf_counter()
        output = model(train_features)
        forward_time += perf_counter() - t_forward

        # Cross Entropy time
        t_CE = perf_counter()
        # loss_train = F.cross_entropy(output, train_labels)

        t_softmax_log = perf_counter()
        softmax_log = F.log_softmax(output,dim=1)
        softmax_time += perf_counter() - t_softmax_log

        t_nll = perf_counter()
        loss_train = F.nll_loss(softmax_log, train_labels)
        nll_time += perf_counter() - t_nll

        cross_entropy_time += perf_counter() - t_CE

        # Backward time
        t_backward = perf_counter()
        loss_train.backward()
        backward_time += perf_counter() - t_backward

        # Step time
        t_step = perf_counter()
        optimizer.step()
        step_time += perf_counter() - t_step

    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time, forward_time, cross_entropy_time, backward_time, step_time, softmax_time, nll_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)

if args.model == "SGC":
    model, acc_val, train_time, forward_time, cross_entropy_time, backward_time, step_time, \
        softmax_time, nll_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])

def print_time_ratio(name, time1, train_time):
    print("{}: {:.4f}s, ratio: {}".format(name, time1, time1/train_time))

def save_time_result(file_name, *args):
    # args is the names of the time
    save_dict = {}
    save_list = []
    for arg in args:
        save_list.append(arg)

    for x in save_list:
        save_dict[x] = eval(x)
    # print(save_dict)
    import pickle
    with open(file_name, 'wb') as f:
        pickle.dump(save_dict, f)


total_time = precompute_time + train_time
print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time,
                                                                              total_time))

print("------Training time Details-------")
print("Total Training Time: {:.4f}s".format(train_time))
# print("Forward time: {:.4f}s, ratio: {}".format(forward_time, forward_time/train_time))
# print("Cross Entropy Time: {:.4f}s, ".format(cross_entropy_time))
# print("Backward Time: {:.4f}s".format(backward_time))
# print("Step Time: {:.4f}s".format(step_time))
print_time_ratio('Forward Time', forward_time, train_time)
print_time_ratio('Cross Entropy Time', cross_entropy_time, train_time)
print("--Cross Entropy Time Details--")
print_time_ratio('Softmax_log Time', softmax_time, train_time)
print_time_ratio('NLL Time', nll_time, train_time)
print_time_ratio('Backward Time', backward_time, train_time)
print_time_ratio('Step Time', step_time, train_time)

file_name = os.path.join('time_result', args.dataset)
save_time_result(file_name, 'total_time', 'precompute_time', 'train_time', 'forward_time', 'cross_entropy_time', 'softmax_time',
                 'nll_time','backward_time', 'step_time')

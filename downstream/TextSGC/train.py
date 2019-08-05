import time
import argparse
import numpy as np
import pickle
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import tabulate
from functools import partial
from utils import *
from models import SGC

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20ng', help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='training batch size.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')
parser.add_argument('--preprocessed', action='store_true',
                    help='use preprocessed data')
parser.add_argument('--optimizer', type=str, default='LBFGS', help='optimizer')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'
if args.tuned:
    with open("tuned_result/{}.SGC.tuning.txt".format(args.dataset), "r") as f:
        args.weight_decay = float(f.read())

torch.backends.cudnn.benchmark = True
set_seed(args.seed, args.cuda)

sp_adj, index_dict, label_dict = load_corpus(args.dataset)
for k, v in label_dict.items():
    if args.dataset == "mr":
        label_dict[k] = torch.Tensor(v).to(args.device)
    else:
        label_dict[k] = torch.LongTensor(v).to(args.device)
features = torch.arange(sp_adj.shape[0]).to(args.device)

adj = sparse_to_torch_sparse(sp_adj, device=args.device)


def train_linear(model, feat_dict, weight_decay, binary=False):
    if not binary:
        act = partial(F.log_softmax, dim=1)
        criterion = F.nll_loss
    else:
        act = torch.sigmoid
        criterion = F.binary_cross_entropy
    if args.optimizer == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.1,
                               weight_decay=5e-6)

    best_val_loss = float('inf')
    best_val_acc = 0
    plateau = 0

    forward_time = 0
    cross_entropy_time = 0
    backward_time = 0
    step_time = 0
    softmax_time = 0
    nll_time = 0

    start = time.perf_counter()
    if args.optimizer == 'LBFGS':
        for epoch in range(args.epochs):
            def closure():
                optimizer.zero_grad()
                # output = model(feat_dict["train"].cuda()).squeeze()
                output = model(feat_dict["train"]).squeeze()
                l2_reg = 0.5*weight_decay*(model.W.weight**2).sum()
                # loss = criterion(act(output), label_dict["train"].cuda())+l2_reg
                loss = criterion(act(output), label_dict["train"])+l2_reg
                loss.backward()
                return loss

            optimizer.step(closure)
    else:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            t_forward = perf_counter()
            output = model(feat_dict["train"]).squeeze()
            forward_time += perf_counter() - t_forward

            t_cross_entropy = perf_counter()
            l2_reg = 0.5 * weight_decay * (model.W.weight ** 2).sum()
            t_softmax = perf_counter()
            softmax_log = act(output)
            softmax_time += perf_counter() - t_softmax

            t_nll = perf_counter()
            loss = criterion(softmax_log, label_dict["train"])+l2_reg
            nll_time += perf_counter() - t_nll
            cross_entropy_time += perf_counter() - t_cross_entropy

            t_backward = perf_counter()
            loss.backward()
            backward_time += perf_counter() - t_backward

            t_step = perf_counter()
            optimizer.step()
            step_time += perf_counter() - t_step

    train_time = time.perf_counter()-start
    # val_res = eval_linear(model, feat_dict["val"].cuda(),
    #                       label_dict["val"].cuda(), binary)
    val_res = eval_linear(model, feat_dict["val"], label_dict["val"], binary)
    if args.optimizer == 'LBFGS':
        return val_res['accuracy'], model, train_time
    else:
        return val_res['accuracy'], model, train_time, forward_time, cross_entropy_time, backward_time, step_time, softmax_time, nll_time

def eval_linear(model, features, label, binary=False):
    model.eval()
    if not binary:
        act = partial(F.log_softmax, dim=1)
        criterion = F.nll_loss
    else:
        act = torch.sigmoid
        criterion = F.binary_cross_entropy

    with torch.no_grad():
        output = model(features).squeeze()
        loss = criterion(act(output), label)
        if not binary: predict_class = output.max(1)[1]
        else: predict_class = act(output).gt(0.5).float()
        correct = torch.eq(predict_class, label).long().sum().item()
        acc = correct/predict_class.size(0)

    return {
        'loss': loss.item(),
        'accuracy': acc
    }
if __name__ == '__main__':
    if args.dataset == "mr": nclass = 1
    else: nclass = label_dict["train"].max().item()+1
    if not args.preprocessed:
        adj_dense = sparse_to_torch_dense(sp_adj, device='cpu')
        feat_dict, precompute_time = sgc_precompute(adj, adj_dense, args.degree-1, index_dict)
    else:
        # load the relased degree 2 features
        with open(os.path.join("preprocessed",
            "{}.pkl".format(args.dataset)), "rb") as prep:
            feat_dict = pkl.load(prep)
        precompute_time = 0

    model = SGC(nfeat=feat_dict["train"].size(1),
                nclass=nclass)
    if args.cuda: model.cuda()

    if args.optimizer == 'LBFGS':
        val_acc, best_model, train_time = train_linear(model, feat_dict, args.weight_decay, args.dataset=="mr")
    else:
        val_acc, best_model, train_time, forward_time, cross_entropy_time, backward_time, \
            step_time, softmax_time, nll_time = train_linear(model, feat_dict, args.weight_decay, args.dataset=="mr")
    # test_res = eval_linear(best_model, feat_dict["test"].cuda(),
    #                        label_dict["test"].cuda(), args.dataset=="mr")
    # train_res = eval_linear(best_model, feat_dict["train"].cuda(),
    #                         label_dict["train"].cuda(), args.dataset=="mr")
    test_res = eval_linear(best_model, feat_dict["test"],
                           label_dict["test"], args.dataset=="mr")
    train_res = eval_linear(best_model, feat_dict["train"],
                            label_dict["train"], args.dataset=="mr")
    print("Total Time: {:2f}s, Train acc: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}".format(precompute_time+train_time, train_res["accuracy"], val_acc, test_res["accuracy"]))
    print("Precompute Time: {:2f}s, Train Time: {:2f}s".format(precompute_time, train_time))

    total_time = precompute_time + train_time

    if args.optimizer != 'LBFGS':
        print_time_ratio('Forward Time', forward_time, train_time)
        print_time_ratio('Cross Entropy Time', cross_entropy_time, train_time)
        print("--Cross Entropy Time Details--")
        print_time_ratio('Softmax_log Time', softmax_time, train_time)
        print_time_ratio('NLL Time', nll_time, train_time)
        print_time_ratio('Backward Time', backward_time, train_time)
        print_time_ratio('Step Time', step_time, train_time)

        file_name = os.path.join('time_result', args.dataset)
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

        save_time_result(file_name, 'total_time', 'precompute_time', 'train_time', 'forward_time', 'cross_entropy_time',
                         'softmax_time',
                         'nll_time', 'backward_time', 'step_time')

        #
        # file_name = os.path.join('time_result', 'reddit')
        # save_time_result(file_name, 'total_time', 'precompute_time', 'train_time', 'forward_time', 'cross_entropy_time',
        #                  'softmax_time',
        #                  'nll_time', 'backward_time', 'step_time')
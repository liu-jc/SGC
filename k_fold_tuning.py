import time
import argparse
import numpy as np
import pickle as pkl
import os
from math import log
from citation import train_regression, test_regression
from models import get_model
from utils import sgc_precompute, load_citation, set_seed, rw_restart_precompute, test_split
from args import get_citation_args
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def sgc_objective(space):
    acc_val_list = []
    for idxs in idx_splits:
        idx_train, idx_val, _ = idxs['train_idx'], idxs['val_idx'], idxs['test_idx']
        model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)
        model, cur_acc_val, _ = train_regression(model, features[idx_train], labels[idx_train], features[idx_val],
                                             labels[idx_val],
                                             args.epochs, space['weight_decay'], args.lr, args.dropout)
        acc_val_list.append(cur_acc_val)
    acc_val = np.average(acc_val_list)
    print('weight decay: {:.2e} '.format(space['weight_decay']) + 'accuracy: {:.4f}'.format(acc_val))
    return {'loss': -acc_val, 'status': STATUS_OK}

# Arguments
args = get_citation_args()

# setting random seeds
set_seed(args.seed, args.cuda)

# Hyperparameter optimization
# space = {'weight_decay': hp.choice('weight_decay', [1e-6,1e-5,1e-4,1e-3])}
space = {'weight_decay': hp.loguniform('weight_decay', log(1e-10), log(1e-4))}
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda, gamma=args.gamma)
idx_splits = test_split(args.dataset)
idx_splits.append({'train_idx': idx_train, 'val_idx': idx_val, 'test_idx': idx_test})

if args.model == "SGC" and args.normalization == "RWalkRestart":
    alpha_space = [0.05,0.1,0.15,0.2,0.25]
    results = []
    for alpha in alpha_space:
        print("alpha is {}".format(alpha))
        features, precompute_time = rw_restart_precompute(features, adj, args.degree, alpha)
        best = fmin(sgc_objective, space=space, algo=tpe.suggest, max_evals=200)
        print("alpha: {}, best: {}".format(alpha, best))

        model = get_model(args.model, features.size(1), labels.max().item() + 1, args.hidden, args.dropout, args.cuda)
        model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train],
                                                      features[idx_val], labels[idx_val],
                                                      args.epochs, best["weight_decay"], args.lr, args.dropout)

        acc_test_list = []
        for idxs in idx_splits:
            acc_test_list.append(test_regression(model, features[idx_test], labels[idx_test]))

        print("Test Accuracy: {:.4f}".format(np.average(acc_test_list)))
        results.append({'alpha':alpha, 'weight_decay':best['weight_decay'], 'test_acc':np.average(acc_test_list)})
    os.makedirs("./results",exist_ok=True)
    with open('./results/{}-{}-{}.txt'.format(args.dataset, args.model, args.normalization),'w') as f:
        for result in results:
            f.write(str(result))
    with open('./results/{}-{}-{}.pkl'.format(args.dataset, args.model, args.normalization),'wb') as f:
        pkl.dump(results,f)
elif args.model == "SGC" and args.normalization == "RWalkRestartS":
    alpha_space = [0.05,0.1,0.15,0.2,0.25]
    results = []
    for alpha in alpha_space:
        print("alpha is {}".format(alpha))
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda,
                                                                            gamma=args.gamma, alpha=alpha)
        idx_splits = test_split(args.dataset)
        idx_splits.append({'train_idx': idx_train, 'val_idx': idx_val, 'test_idx': idx_test})

        features, precompute_time = sgc_precompute(features, adj, args.degree, args.concat)
        best = fmin(sgc_objective, space=space, algo=tpe.suggest, max_evals=200)
        print("alpha: {}, best: {}".format(alpha, best))

        model = get_model(args.model, features.size(1), labels.max().item() + 1, args.hidden, args.dropout, args.cuda)
        model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train],
                                                      features[idx_val], labels[idx_val],
                                                      args.epochs, best["weight_decay"], args.lr, args.dropout)

        acc_test_list = []
        for idxs in idx_splits:
            acc_test_list.append(test_regression(model, features[idx_test], labels[idx_test]))

        print("Test Accuracy: {:.4f}".format(np.average(acc_test_list)))
        results.append({'alpha':alpha, 'weight_decay':best['weight_decay'], 'test_acc':np.average(acc_test_list)})
    os.makedirs("./results",exist_ok=True)
    with open('./results/{}-{}-{}.txt'.format(args.dataset, args.model, args.normalization),'w') as f:
        for result in results:
            f.write(str(result))
    with open('./results/{}-{}-{}.pkl'.format(args.dataset, args.model, args.normalization),'wb') as f:
        pkl.dump(results,f)
elif args.model == "SGC":
    features, precompute_time = sgc_precompute(features, adj, args.degree, args.concat)
    best = fmin(sgc_objective, space=space, algo=tpe.suggest, max_evals=200)
    print("best: ", best)
    ## test
    model = get_model(args.model, features.size(1), labels.max().item() + 1, args.hidden, args.dropout, args.cuda)
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train],
                                                  features[idx_val], labels[idx_val],
                                                  args.epochs, best["weight_decay"], args.lr, args.dropout)

    acc_test_list = []
    for idxs in idx_splits:
        acc_test_list.append(test_regression(model, features[idx_test], labels[idx_test]))

    print("Test Accuracy: {:.4f}".format(np.average(acc_test_list)))
    result = {'weight_decay': best['weight_decay'], 'test_acc': np.average(acc_test_list)}
    os.makedirs("./results",exist_ok=True)
    with open('./results/{}-{}-{}.txt'.format(args.dataset, args.model, args.normalization),'w') as f:
        f.write(str(result))
    with open('./results/{}-{}-{}.pkl'.format(args.dataset, args.model, args.normalization),'wb') as f:
        pkl.dump(result,f)

os.makedirs("./{}-tuning".format(args.model), exist_ok=True)
path = '{}-tuning/{}-{}.txt'.format(args.model, args.normalization, args.dataset)
with open(path, 'wb') as f: pkl.dump(best, f)


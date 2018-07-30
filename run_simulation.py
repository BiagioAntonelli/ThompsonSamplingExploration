import numpy as np
import pandas as pd

import pickle
import os


from agent.bayesianNN import *
from agent.gtNN import *
from agent.gtLR import *
from agent.bayesianLR import *
from agent.bayesianLR_exploit import *
from agent.vanillaNN import *
from agent.dropoutNN import *
from agent.concretedropoutNN import *

from simulation.simulator import *

from models.bnn import *
from models.blr import *
from models.concrete import *

import argparse


if __name__ == "__main__":

    # Load data
    with open('./data/y_train_red.pickle', 'rb') as handle:
        y_train = pickle.load(handle)
    with open('./data/x_train_red.pickle', 'rb') as handle:
        x_train = pickle.load(handle)
    with open('./data/w_gt_2.pickle', 'rb') as handle:
        w_gt = pickle.load(handle)
    with open('./data/w_start.pickle', 'rb') as handle:
        w_start = pickle.load(handle)
    with open('./data/y_test.pickle', 'rb') as handle:
        y_test = pickle.load(handle)
    with open('./data/x_test.pickle', 'rb') as handle:
        x_test = pickle.load(handle)
    with open('./data/X_sim_big.pickle', 'rb') as handle:
        x_sim = pickle.load(handle)


    agents = {}
    agents["bayesianNN"] = BayesianNN(x_train.shape[1])
    agents["bayesianLR-1"] = BayesianLR(x_train.shape[1]+1,1)
    agents["bayesianLR-05"] = BayesianLR(x_train.shape[1]+1, 0.5)
    agents["bayesianLR_exploit"] = BayesianLR_exploit(x_train.shape[1]+1)
    agents["vanillaNN"] = VanillaNN(0.2)
    agents["dropoutNN"] = DropoutNN(0.2)
    agents["concretedropoutNN"] = ConcretedropoutNN()
 

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=0, type=int)
    parser.add_argument('--n_train', default=500, type=int)
    parser.add_argument('--len_sim', default=2000, type=int)
    parser.add_argument('--n_ads_sel', default=100, type=int)
    parser.add_argument('--n_new_ads', default=1000, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--dynamics', default="False", type=str)
    parser.add_argument('--deep_gt', default="False", type=str)
    parser.add_argument('--reset_train', default="False", type=str)
    parser.add_argument('--freq', default=1, type=int)
    parser.add_argument('--checkpoint', default="False", type=str)
    parser.add_argument('--agent', default="bayesianNN", type=str)
    args = parser.parse_args()

    # Initial training data
    x_start = x_train[0:args.n_train]
    y_start = y_train[0:args.n_train]

    model = agents[args.agent]
    exp = args.exp

    path = "./experiments/sim_"+str(exp)
    filename = path+"/"+str(exp)+'checkpoint.pickle'

    if args.checkpoint == "True" and os.path.exists(filename):
            with open(filename, 'rb') as handle:
                regret,r2, x,y,i = pickle.load(handle)

    else:
        regret, r2, x, y,i =  [],[], x_start, y_start, 0


    if args.deep_gt == "True":
        gt_model = gtNN(0.2)
        gt_model.load_state_dict(torch.load("./data/gt_weights_2.pt"))
    else:
        gt_model = gtLR(w_gt)


    _ = simulator().run( model, gt_model, x_sim, x, y, x_test, y_test, args.len_sim, args.n_ads_sel, args.freq,
                  args.n_new_ads, 10,  args.exp, regret, r2,i)

    clicks, regret, r2_score = _

    path = "./experiments/sim_"+ str(exp)
    try:
        os.mkdir(path)
    except Exception:
        None

    with open(path+"/"+str(exp)+'.pickle', 'wb') as handle:
        pickle.dump(_, handle, protocol=pickle.HIGHEST_PROTOCOL)

import os
import pickle
from sklearn.metrics import log_loss, r2_score
from sklearn import preprocessing
import math
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class simulator:

    def __init__(self):
        self.n_adsel = 0

    def run(self, model, gt_model, x_sim, x, y, x_test, y_test, len_sim=1000, n_adsel=100, updates_freq=1,
            n_new=100, seed=10, exp=0, regret=[], r2=[],start_iter = 0):

        clicks = []
        x,y = np.array(x),np.array(y)
        self.n_adsel = n_adsel

        for i in range(start_iter, len_sim):
            np.random.seed(i)
            idx = (i*n_new) % len(x_sim)
            x_new = x_sim.iloc[idx:idx+n_new, :]

            if i % updates_freq == 0:
                model.reset()
                model.train_epoch(x,y)
                y_hat_test = model.predict(x_test)
                ll_ts = log_loss(y_test, y_hat_test)
                print("test logloss: ", ll_ts)

            # Generate outcomes
            rev, n_clicks, x_batch, y_batch, p_pool,p_pool_true = self.update_simulation(model, gt_model, x_new)
            x = np.vstack([x, x_batch])
            y = np.hstack([y, y_batch])

            # Ground Truth
            rev_gt, n_clicks_gt, x_batch_gt, y_batch_gt, p_pool_gt, p_pool_true_gt = self.update_simulation(gt_model, gt_model, x_new)

            # store outcomes
            clicks.append(n_clicks)
            regret.append(rev_gt - rev)
            r2.append(r2_score(p_pool_true, p_pool))

            # print regret
            if i % 20 == 0:
                print("day "+str(i)+"...")
                print("regret  = ", rev_gt - rev)

            # save checkpoints
            if i % 200 == 0:
                _ = regret, r2, x, y, i
                path = "./experiments/sim_"+str(exp)
                os.makedirs(path, exist_ok=True)
                with open(path+"/"+str(exp)+'checkpoint.pickle', 'wb') as handle:
                    pickle.dump(_, handle, protocol=pickle.HIGHEST_PROTOCOL)
   

        return clicks, regret, r2

    def update_simulation(self, net, gt_net, X_new):
        
        # selected from model
        p_new = net.sample_prediction(X_new).squeeze()
        _, p_new_true = self.gen_click(X_new, gt_net) # true probability of selected
        selected = np.argsort(-p_new)[:self.n_adsel]
        x_batch = np.array(X_new.iloc[selected, :])
        clicks, p_clicks = self.gen_click(x_batch, gt_net)
        revenue = np.sum(p_clicks)
        return revenue, np.sum(clicks), x_batch, np.squeeze(clicks), p_new, p_new_true

    def gen_click(self, x,gt_model):

        p = np.squeeze(gt_model.predict(x))
        click = [np.random.binomial(1, p[i], 1) for i in range(len(p))]

        return np.array(click), p

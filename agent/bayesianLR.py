import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from models.blr import *

class BayesianLR:
    def __init__(self,D = 361 ,alpha = 1):
        self.H = np.diag(np.ones(D))
        self.mm = np.zeros(D)
        self.alpha = alpha
        self.D = D

    def train_epoch(self, x_train, y_train):
        x_train_bias = np.hstack([x_train, np.ones([x_train.shape[0], 1])])
        self.mm, self.H = fit_bayes_logistic(np.squeeze(y_train), x_train_bias, self.mm, self.H)

    def predict(self, x):
        x_bias = np.hstack([x, np.ones([x.shape[0], 1])])
        return sigmoid(np.dot(x_bias, self.mm.T))

    def sample_prediction(self, x):
        V = np.linalg.inv(self.H)*self.alpha**2 
        w_sample = np.random.multivariate_normal(self.mm, V)
        x_bias = np.hstack([x, np.ones([x.shape[0], 1])])
        return sigmoid(np.dot(x_bias, w_sample.T))

    def reset(self):
        D = self.D
        self.H = np.diag(np.ones(D))
        self.mm = np.zeros(D)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

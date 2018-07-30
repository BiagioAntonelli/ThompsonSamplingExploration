# Credit: https://gist.github.com/vvanirudh/9e30b2f908e801da1bd789f4ce3e7aac
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

import os
print(os.getcwd())

from models.bnn import *

class BayesianNN:
    def __init__(self,n_input):
        self.encoder = preprocessing.OneHotEncoder(sparse=False).fit( np.array([0,1]).reshape(-1,1) )
        self.sigma_prior = np.log(1+np.exp(-5))
        self.n_samples = 1
        self.learning_rate = 1e-4
        self.n_epochs = 1
        self.batch_size = 64
        self.sigma1 = 1/np.exp(0)
        self.sigma2 = 1/np.exp(6)
        self.pi = 0.75
        self.n_input = n_input
        self.model = MLP(n_input, self.sigma_prior, self.n_samples, self.sigma1, self.sigma2, self.pi)

    def train_epoch(self,x_train,y_train,epochs=5):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print(y_train.shape)
        y_train = self.encoder.transform(y_train.reshape(-1,1))
        M = x_train.shape[0]
        n_batches = math.ceil(M / float(self.batch_size))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        batch_size = self.batch_size

        for epoch in range(0, epochs):
            for b in range(0, n_batches):
                self.model.zero_grad()
                X_ = torch.from_numpy(np.array(x_train[b * batch_size: (b+1) * batch_size]).astype('float')).float()
                y_ = torch.from_numpy(np.array(y_train[b * batch_size: (b+1) * batch_size]).astype('float')).float()
                log_pw, log_qw, log_likelihood = self.model.forward_pass_samples(X_, y_)
                loss = criterion(log_pw, log_qw, log_likelihood,n_batches, batch_size)
                loss.backward()
                optimizer.step()


    def predict(self,x):
        x = Variable(torch.from_numpy(np.array(x).astype('float')).float())
        return self.model(x,infer=True).data.numpy()[:, 1]

    def sample_prediction(self,x):
        x = Variable(torch.from_numpy(np.array(x).astype('float')).float())
        return self.model(x,infer=False).data.numpy()[:, 1]


    def reset(self):
        self.model = MLP(self.n_input, self.sigma_prior,self.n_samples, self.sigma1, self.sigma2, self.pi)

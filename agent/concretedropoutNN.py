import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from sklearn import preprocessing
import sys
gpu_id = 0

from models.concrete import *

class ConcretedropoutNN(nn.Module):

    def __init__(self):   
        super(ConcretedropoutNN, self).__init__()
        self.batch_size = 64
        self.l = 1e-4  # Lengthscale
        self.nb_features = 100 #number of hidden neurons
        self.model = None
        self.D = 2  # One mean, one log_var
        self.encoder = preprocessing.OneHotEncoder(sparse=False).fit(np.array([0, 1]).reshape(-1, 1))

    def predict(self, x):
        x = Variable(torch.FloatTensor(x))
        return self.model(x)[0][:, 1].data.numpy() #MC prediction

    def sample_prediction(self, x):
        x = Variable(torch.FloatTensor(x))
        return self.model(x)[0][:, 1].data.numpy()  # MC prediction

    def train_epoch(self,X, Y, epochs = 3):
        Y = self.encoder.transform(Y.reshape(-1, 1))
        N = X.shape[0]
        wr = self.l**2. / N
        dr = 2. / N
        Q = X.shape[1]
        self.model = ConcreteModel(wr, dr, Q, self.nb_features, self.batch_size, self.D)
        #model = model.cuda(gpu_id)
        optimizer = optim.Adam(self.model.parameters())

        for i in range(epochs):
            print("epoch ", i)
            old_batch = 0
            for batch in range(int(np.ceil(X.shape[0]/self.batch_size))):
                batch = (batch + 1)     
                print(old_batch)
                print(self.batch_size*batch)
                _x = X[old_batch: self.batch_size*batch]
                _y = Y[old_batch: self.batch_size*batch]
                x = Variable(torch.FloatTensor(_x))  # .cuda(gpu_id)
                y = Variable(torch.FloatTensor(_y))  # .cuda(gpu_id)
                mean, log_var = self.model(x)
                loss = self.model.heteroscedastic_loss(y, mean, log_var) + self.model.regularisation_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                old_batch += self.batch_size # this in the author implementation is wrong...
                print("here")


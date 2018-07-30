import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define Net

class VanillaNN(nn.Module):

    def __init__(self, p):
        super(VanillaNN, self).__init__()
        self.fc1 = nn.Linear(390, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        self.p = p

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=self.p, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.p, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=1)   
        return x

    def predict(self, x):
        x = Variable(torch.from_numpy(np.array(x).astype('float')).float())
        return torch.exp(self.eval()(x))[:,1].data.numpy()

    def sample_prediction(self, x):
        x = Variable(torch.from_numpy(np.array(x).astype('float')).float())
        return torch.exp(self.eval()(x))[:, 1].data.numpy()

    def train_epoch(self, X, Y, epochs=3, batch_size=64):
        X = torch.from_numpy(np.array(X).astype('float')).float()
        Y = torch.from_numpy(np.array(Y).astype('float')).float()

        opt = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()
        self.train()
        for e in range(epochs):
            for beg_i in range(0, X.shape[0], batch_size):
                x_batch = X[beg_i:beg_i + batch_size, :]
                y_batch = Y[beg_i:beg_i + batch_size]
                x_batch = Variable(x_batch)
                y_batch = Variable(y_batch).long()  # .long()
                opt.zero_grad()
                y_hat = self.train()(x_batch)
                loss = criterion(y_hat, y_batch)
                loss.backward()
                opt.step()


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class gtNN(nn.Module):

    def __init__(self, p):
        super(gtNN, self).__init__()
        self.fc1 = nn.Linear(390, 130)
        self.fc2 = nn.Linear(130, 130)
        self.fc3 = nn.Linear(130, 1)
        self.p = p

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=self.p, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.p, training=self.training)
        x = F.dropout(self.fc3(x), p=self.p, training=self.training)
        x = F.sigmoid(x)  # softmax
        return x

    def predict(self, x):
        x = Variable(torch.from_numpy(np.array(x).astype('float')).float())
        return self.eval()(x).data.numpy()

    def sample_prediction(self, x):
        x = Variable(torch.from_numpy(np.array(x).astype('float')).float())
        return self.eval()(x).data.numpy()

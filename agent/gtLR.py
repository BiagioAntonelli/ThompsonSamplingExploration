import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class gtLR(nn.Module):

    def __init__(self, w_gt):
        self.w_gt = w_gt

    def predict(self, x):
        x_bias = np.hstack([x, np.ones([x.shape[0], 1])])
        return sigmoid(np.dot(x_bias, self.w_gt.T))

    def sample_prediction(self, x):
        x_bias = np.hstack([x, np.ones([x.shape[0], 1])])
        return sigmoid(np.dot(x_bias, self.w_gt.T))

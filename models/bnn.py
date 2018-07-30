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


def gaussian(x, mu, sigma):
    scaling = 1.0 / torch.sqrt(2.0 * np.pi * (sigma ** 2))
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return scaling * bell


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi)) - torch.log(torch.abs(sigma)) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


def scale_mixture_prior(x, sigma1, sigma2, pi):
    first_gaussian = pi * gaussian(x, 0., sigma1)
    second_gaussian = (1 - pi) * gaussian(x, 0., sigma2)
    return torch.log(first_gaussian + second_gaussian)


def log_softmax_likelihood(yhat_softmax, y):
    return torch.sum(y * torch.log(yhat_softmax), 0)


class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior, sigma1, sigma2, pi):
        super(MLPLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = torch.tensor(sigma_prior).float()
        self.sigma1 = torch.tensor(sigma1).float()
        self.sigma2 = torch.tensor(sigma2).float()
        self.pi = torch.tensor(pi).float()
        self.W_mu = nn.Parameter(torch.Tensor(
            n_input, n_output).normal_(0, 0.1))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0, 0.1))
        self.W_ro = nn.Parameter(torch.zeros(n_input, n_output) - 5)
        self.b_ro = nn.Parameter(torch.zeros(n_output)-5)
        self.lpw = 0
        self.lqw = 0

    def forward(self, X, infer=False):  # , online = False):
        if infer:
            output = torch.mm(X, self.W_mu) + \
                self.b_mu.expand(X.size()[0], self.n_output)
            return output

        epsilon_W, epsilon_b = self.get_random()
        W = self.W_mu + torch.log(1 + torch.exp(self.W_ro)) * epsilon_W
        b = self.b_mu + torch.log(1 + torch.exp(self.b_ro)) * epsilon_b  # (2)
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)

        self.lpw = scale_mixture_prior(W, self.sigma1, self.sigma2, self.pi).sum(
        ) + scale_mixture_prior(b, self.sigma1, self.sigma2, self.pi).sum()
        self.lqw = log_gaussian(W, self.W_mu, torch.log(1 + torch.exp(self.W_ro))).sum(
        ) + log_gaussian_logsigma(b, self.b_mu, torch.log(1 + torch.exp(self.b_ro))).sum()

        return output

    def get_random(self):
        return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, 1)), Variable(torch.Tensor(self.n_output).normal_(0, self.sigma_prior))


class MLP(nn.Module):
    def __init__(self, n_input, sigma_prior, n_samples, sigma1, sigma2, pi):
        super(MLP, self).__init__()
        self.l1 = MLPLayer(n_input, 100, sigma_prior, sigma1, sigma2, pi)
        self.l1_relu = nn.ReLU()
        self.l2 = MLPLayer(100, 100, sigma_prior, sigma1, sigma2, pi)
        self.l2_relu = nn.ReLU()
        self.l3 = MLPLayer(100, 2, sigma_prior, sigma1, sigma2, pi)
        self.l3_softmax = nn.Softmax()  # Softmax()#nn.LogSoftmax(dim=1)#
        self.sigma_prior = torch.tensor(sigma_prior).float()
        self.n_samples = n_samples

    def forward(self, X, infer=False, online=False):
        output = self.l1_relu(self.l1(X, infer))  # ,online))
        output = self.l2_relu(self.l2(output, infer))  # ,online))
        output = self.l3_softmax(self.l3(output, infer))  # ,online))
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw
        return lpw, lqw

    def forward_pass_samples(self, X, y):  # ,online=False):
        s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
        for _ in range(self.n_samples):
            output = self(X)
            sample_log_pw, sample_log_qw = self.get_lpw_lqw()
            sample_log_likelihood = log_softmax_likelihood(output, y)
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood

        return s_log_pw/self.n_samples, s_log_qw/self.n_samples, s_log_likelihood/self.n_samples


def criterion(l_pw, l_qw, l_likelihood, n_batches, batch_size):
    return (1./n_batches) * (l_qw - l_pw) - l_likelihood.sum()

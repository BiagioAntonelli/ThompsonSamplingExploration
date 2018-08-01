import sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticLR:
    def __init__(self):
        self.model = LogisticRegression()
        

    def train_epoch(self, x_train, y_train):
        x_train_bias = np.hstack([x_train, np.ones([x_train.shape[0], 1])])
        self.model.fit(x_train_bias, y_train)

    def predict(self, x):
        x_bias = np.hstack([x, np.ones([x.shape[0], 1])])
        return self.model.predict_proba(x_bias)[:,1]

    def sample_prediction(self, x):
        x_bias = np.hstack([x, np.ones([x.shape[0], 1])])
        return self.model.predict_proba(x_bias)[:,1]

    def reset(self):
        self.model = LogisticRegression()

    
    
    


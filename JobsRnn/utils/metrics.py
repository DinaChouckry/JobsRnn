import torch
import numbers
import numpy as np
from sklearn.metrics import f1_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class F1Score(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y = np.empty([0])
        self.y_hat = np.empty([0])
        self.f1score = 0.0

    def update(self , b_y , b_y_hat):
        self.y = np.append(self.y ,b_y.numpy())
        self.y_hat = np.append(self.y_hat ,b_y_hat.squeeze().numpy())
        self.f1score = f1_score(self.y, self.y_hat , average= 'weighted')
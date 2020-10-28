import torch
from torch import nn
import math
from processMedicalImages import getSingleDataExample
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#https://medium.com/@aakashns/image-classification-using-logistic-regression-in-pytorch-ebb96cc9eb79
#https://github.com/jcreinhold/niftidataset/blob/master/niftidataset/dataset.py


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

    #Denotes the total number of samples
    def __len__(self):
        return len(self.list_IDs)

    #Generates one sample of data
    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        #Load data and get label
        X = torch.from_numpy(getSingleDataExample(ID).flatten())
        y = self.labels[ID]

        return X, y


class logisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(logisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, xb):
        return torch.sigmoid(self.linear(xb))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
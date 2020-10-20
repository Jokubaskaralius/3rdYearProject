import torch
from torch import nn
import math
from processMedicalImages import getSingleDataExample
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


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
        X = getSingleDataExample(ID)
        y = self.labels[ID]

        return X, y


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(57600, 1)

    def forward(self, xb):
        return torch.sigmoid(self.lin(xb))
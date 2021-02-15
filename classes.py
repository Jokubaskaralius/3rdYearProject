import torch
import numpy as np
from torch import nn
import math
import nibabel as nib
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
        try:
            img = nib.load(ID)
            image_data = img.get_fdata(dtype=np.float32)
            image_data = torch.from_numpy(image_data)
            image_data = torch.FloatTensor(image_data)
        except:
            raise ValueError(
                f'Failed to load a processed image.\nImages may not have been processed.'
            )
        X = image_data
        y = self.labels[ID]

        return X, y


class logisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(logisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # output dim 4
        self.act = nn.Sigmoid()

    def forward(self, xb):
        return self.act(self.linear(xb))


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
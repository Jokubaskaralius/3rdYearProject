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


#https://pubmed.ncbi.nlm.nih.gov/23645344/
#https://ieeexplore.ieee.org/document/6149937
#https://link.springer.com/article/10.1186/s40537-019-0263-7?shared-article-renderer
# pyramid kernel https://arxiv.org/pdf/1907.02413.pdf
class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()
        self.conv_layer1 = self._conv3D(1, 32)
        self.conv_layer2 = self._conv3D(32, 64)
        self.fc1 = nn.Linear(406272, 128)
        self.fc2 = nn.Linear(128, 4)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)

    def _conv3D(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2), padding=0),
        )
        return conv_layer

    def _fc(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


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

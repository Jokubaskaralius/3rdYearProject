from typing import Dict, Any
import torch
from torch import nn
from torch import optim


class Selector:
    def __init__(self, hyper_param):
        if not isinstance(hyper_param, dict):
            raise TypeError("Expected dict; got %s" %
                            type(hyper_param).__name__)
        if not hyper_param:
            raise ValueError("Expected %s dict; got empty dict" %
                             os.path.basename(__file__))
        self.model_str = hyper_param["model"]
        self.optimizer_str = hyper_param["optimizer"]
        self.loss_func_str = hyper_param["loss_func"]
        self.lr = hyper_param["learning_rate"]

    def __call__(self):
        model = self.model_select()
        optimizer = self.optimizer_select()
        loss_func = self.loss_func_select()

        return (model, optimizer, loss_func)

    def model_select(self):
        if self.model_str == "LR":
            model = LogisticRegression()
        elif self.model_str == "CNN":
            model = CNNModel()
        else:
            raise ValueError("Expected supported model str; got %s" %
                             self.model_str)
        return model

    def optimizer_select(self):
        model_params = self.model_select().parameters()
        if self.optimizer_str == "ADAM":
            optimizer = optim.Adam(model_params)
        elif self.optimizer_str == "SGD":
            optimizer = optim.SGD(model_params)
        else:
            raise ValueError("Expected supported optimizer str; got %s" %
                             self.optimizer_str)
        return optimizer

    def loss_func_select(self):
        if self.loss_func_str == "CEL":
            loss_func = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Expected supported loss func str; got %s" %
                             self.loss_func_str)
        return loss_func


class LogisticRegression(nn.Module):
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
    def __init__(self):
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

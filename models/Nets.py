#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x
#
#
# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class ClassicNN(nn.Module):
    def __init__(self, d_dim=None, dim1=None, dim2=None, l_dim=None):
        super(ClassicNN, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.d_dim = d_dim
        self.l_dim = l_dim
        self.h1 = nn.Sequential(
           nn.Linear(self.d_dim, self.dim1),
           nn.Tanh(),
        )
        self.h2 = nn.Sequential(
           nn.Linear(self.dim1, self.dim2),
           nn.Tanh(),
        )
        self.o = nn.Sequential(
            nn.Linear(self.dim2, self.l_dim),
        )

    def forward(self, x):
        h1_output = self.h1(x)
        h2_output = self.h2(h1_output)
        class_output = self.o(h2_output)
        return class_output

class scDGN(nn.Module):
    def __init__(self, d_dim=None, dim1=None, dim2=None, dim_label=None, dim_domain=None):
        super(scDGN, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.d_dim = d_dim
        self.dim_label = dim_label
        self.dim_domain = dim_domain
        self.feature_extractor = nn.Sequential(
           nn.Linear(self.d_dim, self.dim1),
           nn.Tanh(),
           nn.Linear(self.dim1, self.dim2),
           nn.Tanh(),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.dim2, self.dim_domain),
            nn.Tanh(),
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(self.dim2, self.dim_label),
        )

    def forward(self, x1, x2=None, mode='train', alpha=1):
        feature = self.feature_extractor(x1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.label_classifier(feature)
        if mode == 'train':
            domain_output1 = self.domain_classifier(reverse_feature)
            feature2 = self.feature_extractor(x2)
            reverse_feature2 = ReverseLayerF.apply(feature2, alpha)
            domain_output2 = self.domain_classifier(reverse_feature2)
            return class_output, domain_output1, domain_output2
        elif mode == 'test':
            return class_output


class CNN(nn.Module):
    def __init__(self, d_dim=None, dim1=None, dim2=None, l_dim=None):
        super(CNN, self).__init__()
        self.d_im = d_dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.l_dim = l_dim
        self.conv1 = nn.Conv2d(self.d_im, self.dim1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.dim1, self.dim2, 2)
        self.fc1 = nn.Linear(self.dim2 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.l_dim)

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.dim2 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, n_kernel=100):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=n_kernel, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(1160,100),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(100, 10),
            nn.Softmax())

    def forward(self, x): # x: batch_size, feature
        x = x.view(x.size()[0], 1, x.size()[1])
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class runLSTM(nn.Module):
    def __init__(self, input_size=None, output_size=None, hidden_dim=None, n_layers=None):
        super(runLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        x = self.fc(out)
        return x
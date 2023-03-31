import glob
import os

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional
import torch.optim as optim
import torch.utils.data

import data as Data

num_channels = 1
dim = 28

#https://github.com/Britefury/self-ensemble-visual-domain-adapt/blob/master/network_architectures.py
class Grey_32_64_128_gp(nn.Module):
    def __init__(self, n_classes=10):
        super(Grey_32_64_128_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        # x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x))))
        x = F.relu(self.conv3_3_bn(self.conv3_3(x)))

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 128)
        x = self.drop1(x)

        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


print('Built network')




def train(network, optimizer, train_loader, epoch):
    train_loss = 0
    classification_criterion = nn.CrossEntropyLoss()

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(Data.device)
        target = target.long().to(Data.device)
        X_sup = torch.autograd.Variable(data)
        y_sup = torch.autograd.Variable(target)

        optimizer.zero_grad()
        network.train(mode=True)

        sup_logits_out = network(X_sup)

        # Supervised classification loss
        clf_loss = classification_criterion(sup_logits_out, y_sup)

        loss_expr = clf_loss

        loss_expr.backward()
        optimizer.step()

        n_samples = X_sup.size()[0]

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), (float(clf_loss.data.cpu().numpy()) * n_samples)))
        train_loss += loss_expr.item()

    return train_loss / len(train_loader)


def test(network, test_loader):
    network.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(Data.device)
            target = target.to(Data.device)
            X_sup = torch.autograd.Variable(data)
            y_sup = torch.autograd.Variable(target)

            correct += f_eval_tgt(network, X_sup, y_sup)
        print('\nTest set Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)



print('Compiled training function')

def f_pred_tgt(net,X_sup):
    net.train(mode=False)
    output = net(X_sup)
    return F.softmax(output, dim=1)

def f_y_pred(net, X_sup):
    y_pred_prob = f_pred_tgt(net, X_sup)
    y_pred = torch.argmax(y_pred_prob, dim=1)
    return y_pred


def f_eval_tgt(net, X_sup, y_sup):
    y_pred = f_y_pred(net, X_sup)
    return y_pred.eq(y_sup.data.view_as(y_pred)).sum()

print('Compiled evaluation function')

random_seed = 1


def run_classifier(domain):
    max_epochs = 200
    learning_rate = 0.005
    epsilon = 1e-3

    train_loader, test_loader = Data.get_data(domain)
    network = Grey_32_64_128_gp(10).to(Data.device)
    params = list(network.parameters())

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    train_losses = []

    epoch = 0
    while True:
        epoch += 1

        if epoch >= max_epochs:
            break

        # Convergance
        if len(train_losses) > 3 and \
                np.abs(train_losses[-1] - train_losses[-2]) <= epsilon and \
                np.abs(train_losses[-2] - train_losses[-3]) <= epsilon:
            break

        train_loss = train(network, optimizer, train_loader, epoch)
        train_losses.append(train_loss)
        test(network, test_loader)

    accuracy = test(network, test_loader)
    torch.save(network.state_dict(), "./classifiers_new/{}_classifier.pt".format(domain))

    return accuracy


if __name__ == '__main__':
    run_classifier('MNIST')
    run_classifier('USPS')
    run_classifier('SVHN')
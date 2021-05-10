# Taken from https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/OthelloNNet.py

import sys
import os
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import datetime

class OthelloNNet(nn.Module):
    def __init__(self):
        # game params
        self.board_x = 7
        self.board_y = 6
        self.action_size = 7

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, stride=1)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, 512*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=0.3, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=0.3, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

### Neural Net loss function implemented via PyTorch
class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):


        value_error = (value.flatten() - y_value.flatten()) ** 2
        policy_error = torch.sum((torch.exp(policy)* 
                                (1e-8 + y_policy.float()).float()), 1)

        print(value)

        total_error = (value_error.view(-1).float() - policy_error).mean()
        return total_error

def train(net, dataset, epoch_start=0, epoch_stop=200, cpu=0):
    """
    Inputs of NN, training data
    """
    torch.manual_seed(cpu)  # for debugging
    cuda = torch.cuda.is_available()  # use GPU if possible
    net.train()  # set NN to train
    evaluator = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.2)

    # TODO: convert gameLoop() data to proper format for training

    # train_set = board_data(dataset)
    # train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    train_loader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        scheduler.step()
        total_loss = 0.0
        losses_per_batch = []
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            if cuda:
                state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()
            optimizer.zero_grad()
            policy_pred, value_pred = net(state)
            loss = evaluator(value, value_pred, policy, policy_pred)
            loss.backward()
            optimizer.step()
            total_loss += loss
            if i % 10 == 0:  # print every 10 mini-batches of size = batch_size
                print('Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (os.getpid(), epoch + 1, (i + 1) * 30, len(dataset), total_loss / 10))
                losses_per_batch.append(total_loss / 10)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch) / len(losses_per_batch))
        print(f"Epoch {epoch} total loss: {total_loss}")

    print('Finished Training')
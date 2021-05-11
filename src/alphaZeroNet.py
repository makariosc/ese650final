# THIS FILE TAKEN FROM https://github.com/geochri/AlphaZero_Chess
# LARGE PARTS OF THIS FILE ARE DIRECTLY FROM THE ABOVE LINK
# We have customized this file to work directly with our own implementation.
# We do not plan on selling, commercializing, yadda legal yadda any part of this code.

#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime

class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx].transpose(2,0,1), self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    """
    CNN called in ChessNet()
    Network arch takes input of 22x8x8 (TBD)-> 22x256x3 -> 
    minibatch w/ normalization -> ReLu
    """
    def __init__(self):
        super(ConvBlock, self).__init__()
        # XXX unused action_size object
        self.conv1 = nn.Conv2d(1, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        # TODO: CHANGE DIMENSION ACCORDING TO OUTPUT OF FEN CONVERSION
        s = s.view(-1, 1, 7, 6)  # batch_size x channels x board_x x board_y
        # s = torch.tensor(s)
        s = self.conv1(s.float())
        s = self.bn1(s)
        s = F.relu(s)
        
        
        # s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    """
    Called in ChessNet()
    Used to initialize 19 residual layers
    One ResBlock arch: Conv -> Batch Norm -> ReLu -> Conv -> Batch Norm
    """
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    """
    Output layer called in ChessNet()
    Takes in 2 heads -> value and policy head
    Value head arch: Conv->Batch Norm->ReLu->Linear layer->ReLu->Linear->Tanh
    Policy head arch: Conv->Batch Norm->ReLu->Linear->logsoftmax
    """
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(7*6, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(7*6*128, 7)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 7*6)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 7*6*128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ChessNet(nn.Module):
    """
    Build AlphaZero Deep NN
    """
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(10):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(10):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
        

class AlphaLoss(torch.nn.Module):
    """
    Loss function, mean squared error
    """
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
    
def train(net, dataset, epoch_start=0, epoch_stop=200, cpu=0):
    """
    Inputs of NN, training data
    """
    torch.manual_seed(cpu) # for debugging
    cuda = torch.cuda.is_available() # use GPU if possible
    net.train() # set NN to train
    criterion = AlphaLoss() # initialize loss function
    optimizer = optim.Adam(net.parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)
    
    
    # TODO: convert gameLoop() data to proper format for training
    
    # train_set = board_data(dataset)
    # train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    train_loader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        scheduler.step()
        total_loss = 0.0
        losses_per_batch = []
        for i, data in enumerate(train_loader,0):
            state, policy, value = data
            if cuda:
                state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()
            optimizer.zero_grad()
            policy_pred, value_pred = net(state) # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
            loss = criterion(value_pred[:,0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches of size = batch_size
                print('Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (os.getpid(), epoch + 1, (i + 1)*30, len(dataset), total_loss/10))
                print("Policy:",policy[0].argmax().item(),policy_pred[0].argmax().item())
                print("Value:",value[0].item(),value_pred[0,0].item())
                losses_per_batch.append(total_loss/10)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        if len(losses_per_epoch) > 100:
            if abs(sum(losses_per_epoch[-4:-1])/3-sum(losses_per_epoch[-16:-13])/3) <= 0.01:
                break

    print('Finished Training')

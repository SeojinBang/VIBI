#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
sys.path.append('../')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from return_data import return_data
from utils import str2bool, check_model_name
from pathlib import Path

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    
    model.train()     
    
    for batch_idx, (data, target, _, idx) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()    
        
        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, outfile = False, **kargs):
    
    model.eval()
    test_loss = 0
    correct = 0
    model_name = os.path.splitext(os.path.basename(args.model_name))[0]
    #outfile_path = args.out_dir
    outfile_path = os.path.join(args.data_dir, 'processed')
    outmode = "TEST"

    if outfile: 
        assert kargs['outmode'] in ['train', 'test', 'valid']
        outmode = kargs['outmode']
    
    with torch.no_grad():
        
        predictions = []
        predictions_idx = []
        for data, target, _, idx in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, size_average = False).item() # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if outfile: 
                predictions.extend(pred.data.squeeze(-1).cpu().tolist())
                predictions_idx.extend(idx.cpu().tolist())
                
        if outfile:
            
            predictions = np.array(predictions)
            predictions_idx = np.array(predictions_idx)
            inds = predictions_idx.argsort()
            sorted_predictions = predictions[inds]

            output_name = model_name + '_pred_' + outmode + '.pt'
            torch.save(sorted_predictions, Path(outfile_path).joinpath(output_name))                
            
    test_loss /= len(test_loader.dataset)
    print('\n ', outmode, ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))
    
def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--dataset', default = 'mnist', type = str,
                        help='dataset name: imdb-sent, imdb-word, mnist, mimic')
    parser.add_argument('--data_dir', default = 'dataset', type = str,
                        help='data directory path')
    parser.add_argument('--batch_size', type=int, default = 100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--model_name', default = 'original.ckpt', type = str,
                        help = 'if train is True, model name to be saved, otherwise model name to be loaded')
    parser.add_argument('--epoch', type = int, default=10, metavar='N',
                        help='number of epoch to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--cuda', default = True, type = str2bool,
                        help = 'enable cuda')
    parser.add_argument('--seed', default = 2555, type = int,
                        help='random seed (defalut 2555')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--mode', default = 'train', type=str, help = 'train or test')
    
    args = parser.parse_args()
    
    ## cuda
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    args.cuda = (args.cuda and torch.cuda.is_available()) 
    device = torch.device("cuda" if args.cuda else "cpu")

    ## set random seed
    seed = args.seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed) if args.cuda else None
      
    ## data loader
    args.root = os.path.join(args.data_dir)
    args.load_pred = False
    data_loader = return_data(args)
    train_loader = data_loader['train']
    valid_loader = data_loader['valid']
    test_loader = data_loader['test']
    
    ## define model
    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.5)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    ## fit model        
    if args.mode == 'train' : 

        if 'models' not in os.listdir('.'):
            os.mkdir('models')
            
        model_name = check_model_name(args.model_name)
        file_path = './models'
        model_name = check_model_name(model_name, file_path)      
        
        for epoch in range(1, args.epoch + 1):
            
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, valid_loader, outfile = False) # validation 

        test(args, model, device, train_loader,
             outfile = True, outmode = 'train') # training accuracy
        test(args, model, device, valid_loader,
             outfile = True, outmode = 'valid') # validation accuracy
        test(args, model, device, valid_loader,
             outfile = True, outmode = 'test') # test accuracy
        
        model_name = Path(file_path).joinpath(model_name)
        torch.save(model.state_dict(), model_name)
            
    elif args.mode == 'test' : 
        
        model_name = args.model_name
        file_path = Path('./models')

        assert model_name in os.listdir(file_path)
        
        model_name = Path(file_path).joinpath(model_name)
        model.load_state_dict(torch.load(model_name))
        
        test(args, model, device, test_loader, outfile = True, outmode = 'test') # test accuracy
        
    else :
        
        print('\nError: "--mode train" or "--mode test" expected')
        
if __name__ == '__main__':
    main()  

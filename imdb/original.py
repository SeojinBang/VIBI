#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:31:55 2018

@author: seojin.bang

structure from original model from L2X
"""
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
from utils import str2bool, check_model_name, UnknownModelError, compare_with_previous, TimeDistributed
from pathlib import Path
#%%
class Net(nn.Module):
    
    def __init__(self, **kwargs):

        super(Net, self).__init__()

        self.args = kwargs['args']

        self.model_type = self.args.model_type
        self.max_total_num_words = self.args.max_total_num_words
        self.embedding_dim = self.args.embedding_dim
        self.max_num_words = self.args.max_num_words #100
        self.max_num_sents = self.args.max_num_sents #15  
        self.num_rnn_out_size = 50
        self.num_cnn_hidden_size = 100
        self.num_cnn_kernel_size = 3
        self.bidirectional = True#
        self.mul_bidirect = 2 if self.bidirectional else 1

        ## Model Structure
        self.embedding_layer = nn.Embedding(num_embeddings = self.max_total_num_words + 2, 
                                            embedding_dim = self.embedding_dim) 

        self.dense = nn.Linear(self.max_num_sents * 2 * self.mul_bidirect, 2)

        if self.model_type in ['rnn', 'RNN']:

            self.RNN_word = nn.RNN(input_size = self.args.embedding_dim, hidden_size = self.num_rnn_out_size,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
    
            self.RNN_sent = nn.RNN(input_size = self.num_rnn_out_size * self.mul_bidirect, hidden_size = 2,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
                
            self.model_type = 'rnn'
            self.encoder_word = self.RNN_word
            self.encoder_sent = self.RNN_sent

        elif self.model_type in ['lstm', 'LSTM']:

            self.LSTM_word = nn.LSTM(input_size = self.args.embedding_dim, hidden_size = self.num_rnn_out_size,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
    
            self.LSTM_sent = nn.LSTM(input_size = self.num_rnn_out_size * self.mul_bidirect, hidden_size = 2,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)

            self.model_type = 'lstm'
            self.encoder_word = self.LSTM_word
            self.encoder_sent = self.LSTM_sent
            
        elif self.model_type in ['lstm-light', 'LSTM-light', 'LSTM-Light', 'LSTM-LIGHT']:
            
            self.LSTM_word = nn.LSTM(input_size = self.args.embedding_dim, hidden_size = 1,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)

            self.model_type = 'lstm-light'
            self.encoder_word = self.LSTM_word
            self.dense = nn.Linear(self.max_num_sents * self.max_num_words * self.mul_bidirect, 2)

        elif self.model_type in ['lstm-light-onedirect', 'LSTM-light-onedirect', 'LSTM-Light-onedirect', 'LSTM-LIGHT-onedirect']:
            
            self.LSTM_word = nn.LSTM(input_size = self.args.embedding_dim, hidden_size = 1,
                    num_layers = 1, batch_first = True, bidirectional = False)

            self.model_type = 'lstm-light'
            self.encoder_word = self.LSTM_word
            self.dense = nn.Linear(self.max_num_sents * self.max_num_words, 2)
            
        elif self.model_type in ['gru', 'GRU']:
            
            self.GRU_word = nn.GRU(input_size = self.args.embedding_dim, hidden_size = self.num_rnn_out_size,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
    
            self.GRU_sent = nn.GRU(input_size = self.num_rnn_out_size * self.mul_bidirect, hidden_size = 2,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)

            self.model_type = 'gru'
            self.encoder_word = self.GRU_word
            self.encoder_sent = self.GRU_sent

        elif self.model_type in ['cnn', 'CNN']:
        
            self.model_type = 'cnn'
            self.encoder_word = nn.Sequential(
                    nn.Dropout(p = 0.2),
                    nn.Conv1d(in_channels = 1, out_channels = self.num_cnn_hidden_size, kernel_size = self.embedding_dim * self.num_cnn_kernel_size, padding = 0, stride = self.embedding_dim), # torch.Size([150, 250, 98])
                    nn.ReLU(True),
                    nn.MaxPool1d(kernel_size = self.max_num_words - self.num_cnn_kernel_size + 1)
                    )
            self.encoder_sent = nn.Sequential(
                        nn.Linear(in_features = self.num_cnn_hidden_size, out_features = 2),
                        nn.LogSoftmax(1)
                        )

        else:
            
            raise UnknownModelError()

    def forward(self, x):

        ## word embedding
        embeded_words = self.embedding_layer(x)
        
        pred = self.forward_sub(embeded_words)

        return pred
    
    def forward_sub(self, embeded_words):
        
        if self.model_type in ['rnn', 'lstm', 'gru']:
            
            embeded_words = self.encoder_word(embeded_words.view(-1, self.max_num_words, self.embedding_dim))[0] # batch * sent, word-dim, hidden-dim

            #encoded_sent = embeded_words.contiguous().view(-1, self.max_num_sents, self.max_num_words * self.num_rnn_out_size * self.mul_bidirect)
            encoded_sent = torch.mean(embeded_words, -2).view(-1, self.max_num_sents, self.num_rnn_out_size * self.mul_bidirect)

            encoded_sent = self.encoder_sent(encoded_sent)[0]
            encoded_review = encoded_sent.contiguous().view(-1, self.max_num_sents * 2 * self.mul_bidirect)
            
            encoded_review = self.dense(encoded_review)

            pred = F.log_softmax(encoded_review, dim = -1)
        
        elif self.model_type in ['lstm-light', 'LSTM-light', 'LSTM-Light', 'LSTM-LIGHT']:
            
            embeded_words = self.encoder_word(embeded_words)[0]
            encoded_review = self.dense(embeded_words.contiguous().view(embeded_words.size(0), -1, 1).squeeze(-1))
            pred = F.log_softmax(encoded_review, dim = -1)

        elif self.model_type in ['cnn', 'CNN']:

            #encoded_sent = TimeDistributed(self.encoder_word)(embeded_words) # batch * sent, word, dim -> 1500, 1, 100
            encoded_sent = self.encoder_word(embeded_words.view(-1, 1, self.max_num_words * self.embedding_dim))

            encoded_sent = encoded_sent.view(-1, self.max_num_sents, self.num_cnn_hidden_size) # batch * 15 * 100
            encoded_review = torch.mean(encoded_sent, -2) # batch * 100
            pred = self.encoder_sent(encoded_review) # shoudl be [batch, 2]
            
        else:
            raise UnknownModelError()

        return pred

def train(args, model, device, train_loader, optimizer, epoch):
    
    model.train()     
    correct = 0
    total_size = 0

    for batch_idx, batch in enumerate(train_loader):
#    for batch_idx, (data, target, _, idx) in enumerate(train_loader):

        data = batch.text
        target = batch.label.view(-1) - 2
        data, target = data.to(device), target.to(device)
        total_size += data.size(0)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.max(1, keepdim = True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('TRAIN set (epoch {}): Accuracy: {}/{} ({:.0f}%)'.format(epoch,
        correct, total_size, 100. * correct / total_size))

def test(args, model, device, test_loader, outfile = False, compare = True, **kwargs):
    
    model.eval()
    test_loss = 0
    correct = 0
    correct_pre = 0
    model_name = os.path.splitext(os.path.basename(args.model_name))[0]
    outfile_path = os.path.join(args.data_dir, 'processed')# if yo want to change this directory, you also need to change 'path_to_file' object in 'data_loader.py'
    outmode = 'VALID'

    if outfile:
        
        assert kwargs['outmode'] in ['train', 'test', 'valid']
        outmode = kwargs['outmode']
        #label_pred_dict = torch.load('dataset/processed/original_lstm_pred_' + outmode + '.pt')
        
    with torch.no_grad():
        
        predictions = []
        predictions_idx = []
        total_size = 0

        for batch_idx, batch in enumerate(test_loader):
        #for data, target, _, idx in test_loader:    
            data = batch.text
            target = batch.label.view(-1) - 2
            fname = batch.fname
            target_pred = batch.label_pred.view(-1)#?remove
            
            total_size += data.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_pre += target_pred.eq(target.view_as(target_pred)).sum().item()#?remove
            
            if outfile: 
                predictions.extend(pred.data.squeeze(-1).cpu().tolist())
                predictions_idx.extend(fname.squeeze(0).cpu().tolist())
        
        test_loss /= total_size 
        if outfile:
                
            predictions = np.array(predictions)
            predictions_idx = np.array(predictions_idx)
            #inds = predictions_idx.argsort()
            #sorted_predictions = predictions[inds]

            predictions_all = dict(zip(predictions_idx, predictions))
            
            print(outmode, 'set: from last prediction, Accuracy: {}/{} ({:.0f}%)'.format(correct_pre, total_size, 100. * correct_pre / total_size))#? remove
            print(outmode, 'set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, total_size, 100. * correct / total_size))
            
            compare = False if outmode is not 'valid' else compare
            compare_with_previous(correct, correct_pre, compare = compare)
            
            if not os.path.exists(outfile_path):
                os.mkdir(outfile_path)

            output_name = model_name + '_pred_' + outmode + '.pt'
            torch.save(predictions_all, Path(outfile_path).joinpath(output_name))                
            
        else:
            print(outmode, 'set: from last prediction, Accuracy: {}/{} ({:.0f}%)'.format(correct_pre, total_size, 100. * correct_pre / total_size))
            print(outmode, 'set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, total_size, 100. * correct / total_size))
            
#%%
def main():
#%%
    parser = argparse.ArgumentParser(description='Original model')
    
    parser.add_argument('--dataset', default = 'imdb', type = str, help='dataset name: imdb, mnist, mimic')
    parser.add_argument('--data_dir', default = 'dataset', type = str, help='data directory path')
    parser.add_argument('--batch_size', type=int, default = 50, metavar='N', help='input batch size for training')
    parser.add_argument('--model_name', default = 'original.ckpt', type = str, help = 'if train is True, model name to be saved, otherwise model name to be loaded')
    parser.add_argument('--model_type', default = 'lstm-light', type = str, help = 'cnn, lstm, rnn, gru, lstm-light')
    parser.add_argument('--epoch', type = int, default = 2, metavar='N', help='number of epoch to train')
    parser.add_argument('--lr', type=float, default = 0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--cuda', default = True, type = str2bool, help = 'enable cuda')
    parser.add_argument('--log-interval', type=int, default = 10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--mode', default = 'train', type=str, help = 'train or test')

    args = parser.parse_args()
    
    ## cuda
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    args.cuda = (args.cuda and torch.cuda.is_available()) 
    device = torch.device("cuda" if args.cuda else "cpu")
#%%
#    ## set random seed
#    seed = args.seed
#    torch.manual_seed(seed)
#    if args.cuda:
#        torch.cuda.manual_seed(seed) if args.cuda else None
   
    ## data loader
    args.root = os.path.join(args.data_dir)
    args.load_pred = False
    data_loader = return_data(args)
    train_loader = data_loader['train']
    valid_loader = data_loader['valid']
    test_loader = data_loader['test']

    args.max_total_num_words = data_loader['max_total_num_words']
    args.embedding_dim = data_loader['embedding_dim']
    args.max_num_words = data_loader['max_num_words'] #100
    args.max_num_sents = data_loader['max_num_sents'] #15    
#%%   
    ## define model
    model = Net(args = args).to(device)
    # optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.5)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
#%%
    ## fit model        
    if args.mode == 'train' : 

        if 'models' not in os.listdir('.'):
            os.mkdir('models')

        file_path = './models'
        args.model_name = check_model_name(args.model_name, file_path)      
        
        for epoch in range(1, args.epoch + 1):
            
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, valid_loader, outfile = False) # validation
        
        print()
        test(args, model, device, valid_loader, outfile = True, outmode = 'valid') # validation accuracy
        test(args, model, device, test_loader, outfile = True, outmode = 'test') # test accuracy
        test(args, model, device, train_loader, outfile = True, outmode = 'train') # training accuracy

        model_name = Path(file_path).joinpath(args.model_name)
        torch.save(model.state_dict(), model_name)
            
    elif args.mode == 'test':
        
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

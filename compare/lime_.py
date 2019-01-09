#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:12:32 2018

@author: seojin.bang
"""
import argparse
import os
import sys
import re
import copy
import matplotlib.pyplot as plt
sys.path.append('../')
import numpy as np
from return_data import return_data
#from scipy import misc
#import cv2
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from utils import cuda, str2bool, label2binary, cuda, idxtobool, UnknownDatasetError, index_transfer, save_batch
from pathlib import Path
from lib.image_utils import Segmentation
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torch.nn import functional as F
from sklearn.utils import check_random_state
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression # Ridge
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')
#from torchvision.models import vggz16, vgg19
#from torchvision.utils import save_image
#from lib.gradients import VanillaGrad, SmoothGrad, GuidedBackpropGrad, GuidedBackpropSmoothGrad
#from lib.image_utils import preprocess_image, save_as_gray_image
#from lib.labels import IMAGENET_LABELS
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
#from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm  
from lib.lime_library import LimeImageExplainerModified, LimeTextExplainerModified
from lib.image_utils import Segmentation

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default = 'mnist', type = str, help='dataset name: imdb-sent, imdb-word, mnist, mimic')
    parser.add_argument('--default_dir', default = '.', type = str, help='default directory path')
    parser.add_argument('--data_dir', default = 'dataset', type = str, help='data directory path')
    parser.add_argument('--batch_size', type=int, default = 100, metavar='N', help='input batch size for training (default: 64)')
    #parser.add_argument('--model_name', required = True, type = str, help = 'if train is True, model name to be saved, otherwise model name to be loaded')
    #parser.add_argument('--model_name', default = 'original.ckpt', type = str, help = 'if train is True, model name to be saved, otherwise model name to be loaded')
    parser.add_argument('--model_name', default = 'original.ckpt', type = str, help = 'if train is True, model name to be saved, otherwise model name to be loaded') 
    #parser.add_argument('--chunk_size', default = 1, type = int, help='chunk size. for image, chunk x chunk will be the actual chunk size')
    parser.add_argument('--chunk_size', default = 2, type = int, help='chunk size. for image, chunk x chunk will be the actual chunk size')
    parser.add_argument('--cuda', default = False, type = str2bool, help = 'enable cuda')
    parser.add_argument('--out_dir', type=str, default='./result/lime/', help='Result directory path')
    parser.add_argument('--K', type = int, default = -1, help='dimension of encoding Z')
    parser.add_argument('--chunked', type = str2bool, default = True, help='True is neighborhood samples are fuzzed within each chunk')
    parser.add_argument('--segment_filter_size', type = int, default = 1, help = 'only for imdb-1 for cnn, 10 for lstm')
    args = parser.parse_args()
    
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")
    
    print('Input data: {}'.format(args.dataset))

    return args
#%%  
def main():
#%%      
    args = parse_args()

    if not os.path.exists(args.out_dir):

        os.makedirs(args.out_dir)

    ## Data Loader
    args.root = os.path.join(args.data_dir)
    args.load_pred = False
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.model_dir = '../' + args.dataset + '/models'
    device = torch.device("cuda" if args.cuda else "cpu")
    
    data_loader = return_data(args)
    test_loader = data_loader['test']

    if 'mnist' in args.dataset:
    
        from mnist.original import Net
        
        ## load model
        model = Net().to(device) 
        
        args.word_idx = None
        args.original_ncol = 28
        args.original_nrow = 28
        args.chunk_size = args.chunk_size if args.chunk_size > 0 else 1
        assert np.remainder(args.original_nrow, args.chunk_size) == 0
        args.filter_size = (args.chunk_size, args.chunk_size)
        args.explainer = LimeImageExplainerModified(verbose = False, feature_selection = 'highest_weights', is_cuda = args.cuda, dataset = args.dataset)
        args.explainer_all = LimeImageExplainerModified(verbose = False, feature_selection = 'none', is_cuda = args.cuda, dataset = args.dataset)
        args.segmenter = Segmentation(dataset = args.dataset,
                                      filter_size = (2, 2),
                                      #filter_size = args.filter_size,
                                      is_cuda = args.cuda)
        args.model_regressor = LogisticRegression(random_state = 0, solver='lbfgs', max_iter = 200, fit_intercept=True, multi_class='ovr')
        args.num_samples = 1000
        
    elif 'imdb' in args.dataset:
    
        from imdb.original import Net
        args.word_idx = data_loader['word_idx']
        args.max_total_num_words = data_loader['max_total_num_words']
        args.embedding_dim = data_loader['embedding_dim']
        args.max_num_words = data_loader['max_num_words'] #100
        args.max_num_sents = data_loader['max_num_sents'] #15
        args.model_type = args.model_name.split('_')[-1].split('.')[0]
        
        ## load model
        model = Net(args = args).to(device)
        
        args.original_ncol = args.max_num_words
        args.original_nrow = args.max_num_sents
        args.chunk_size = args.chunk_size if args.chunk_size > 0 else 1
        if args.chunk_size > args.original_ncol: args.chunk_size = args.original_ncol
        args.filter_size = (1, args.chunk_size)
        args.explainer = LimeTextExplainerModified(verbose = False,
                                                   feature_selection = 'highest_weights', is_cuda = args.cuda, dataset = args.dataset)
        args.explainer_all = LimeTextExplainerModified(verbose = False,
                                                       feature_selection = 'none', is_cuda = args.cuda, dataset = args.dataset)
        args.segmenter = Segmentation(dataset = args.dataset,
                                      filter_size = (1, args.segment_filter_size),
                                      is_cuda = args.cuda)
        args.model_regressor = LogisticRegression(random_state = 0,
                                                  solver='lbfgs',
                                                  max_iter = 200,
                                                  fit_intercept=True,
                                                  multi_class='ovr')
        args.num_samples = 100
    else:
    
        raise UnknownDatasetError()

    model_name = Path(args.model_dir).joinpath(args.model_name)
    model.load_state_dict(torch.load(model_name, map_location='cpu'))
#%%
    if args.cuda:
        model.cuda()
#%%  
    ## Prediction
    test(args, model, device, test_loader, k = args.K)
    
#%%    
def test(args, model, device, test_loader, k, **kargs):
    '''
    k: the number of raw features selected
    '''
#%%     
    model.eval()
    test_loss = 0
    total_num = 0
    total_num_ind = 0
    correct = 0
    
    correct_zeropadded = 0
    precision_macro_zeropadded = 0  
    precision_micro_zeropadded = 0
    precision_weighted_zeropadded = 0
    recall_macro_zeropadded = 0
    recall_micro_zeropadded = 0
    recall_weighted_zeropadded = 0
    f1_macro_zeropadded = 0
    f1_micro_zeropadded = 0
    f1_weighted_zeropadded = 0

    vmi_zeropadded_sum = 0
    vmi_fidel_sum = 0
    vmi_fidel_fixed_sum = 0 
    
    correct_approx = 0
    precision_macro_approx = 0
    precision_micro_approx = 0
    precision_weighted_approx = 0
    recall_macro_approx = 0
    recall_micro_approx = 0
    recall_weighted_approx = 0
    f1_macro_approx = 0
    f1_micro_approx = 0
    f1_weighted_approx = 0
    
    correct_approx_fixed = 0
    precision_macro_approx_fixed = 0
    precision_micro_approx_fixed = 0
    precision_weighted_approx_fixed = 0
    recall_macro_approx_fixed = 0
    recall_micro_approx_fixed = 0
    recall_weighted_approx_fixed = 0
    f1_macro_approx_fixed = 0
    f1_micro_approx_fixed = 0
    f1_weighted_approx_fixed = 0    
        
    is_cuda = args.cuda
    j = 0
#%%
    start = time.time()
    for idx, batch in enumerate(test_loader):#(data, target, _, _)
        #print('####### idx break ####### you must remove this before running')
        #if idx > 0:
        #    break
        if 'mnist' in args.dataset:
            
            num_labels = 10
            data = batch[0]
            target = batch[1]
            idx_list = [0, 1, 2, 3]
        
        elif 'imdb' in args.dataset:
            
            num_labels = 2
            data = batch.text
            target = batch.label.view(-1) - 2
            fname = batch.fname
            idx_list = [0, 200]
            
        else:
        
            raise UnknownDatasetError()
                   
        data, target = data.to(device), target.to(device)
        if 'imdb' in args.dataset:
            data_embeded = model.embedding_layer(data) # batch, sent * word * embed
        output_all = model(data)
        test_loss += F.cross_entropy(output_all, target, reduction = 'sum').item()
        pred = output_all.max(1, keepdim=True)[1] # get the index of the max log-prob     
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_num += 1
        #total_num_ind += data.size(0)

        if idx == 0:
            if 'mnist' in args.dataset:
                args.segments = args.segmenter(Variable(data[0:1],
                                                        requires_grad = False))
                n_features = np.unique(args.segments).shape[0]
                args.segments_data = check_random_state(None).randint(0, 2, args.num_samples * n_features).reshape((args.num_samples, n_features))
                args.segments_data[0, :] = 1
                
            elif 'imdb'in args.dataset:
                args.segments = args.segmenter(Variable(data[0:1].view(1, 1, args.max_num_sents, args.max_num_words), requires_grad = False), args.embedding_dim)
                n_features = np.unique(args.segments).shape[0]
                args.segments_data = check_random_state(None).randint(0, 2, args.num_samples * n_features).reshape((args.num_samples, n_features))
                args.segments_data[0, :] = 1

        total_num_sub  = 0
        for i in range(data.size(0)):
            #print('i',i)

            if 'mnist' in args.dataset:
            
                ## Calculate Gradient
                input = Variable(data[i:(i+1)], requires_grad = False)

                #output = model(input)
                #output = torch.max(output)
                explanation = args.explainer.explain_instance(input, 
                                         filter_size = args.filter_size, 
                                         classifier_fn = model, 
                                         top_labels = 10, 
                                         hide_color = 0, 
                                         num_features = k,
                                         num_samples = args.num_samples, 
                                         model_regressor = args.model_regressor,
                                         segments = args.segments,
                                         segments_data = args.segments_data,
                                         chunked = args.chunked)
                explanation_all = args.explainer_all.explain_instance(input, 
                                         filter_size = args.filter_size,
                                         classifier_fn = model, 
                                         top_labels = 10, 
                                         hide_color = 0, 
                                         #num_features = k,
                                         num_samples = args.num_samples, 
                                         model_regressor = args.model_regressor,
                                         segments = args.segments,
                                         segments_data = args.segments_data,
                                         chunked = args.chunked)

                if explanation.local_exp[explanation.top_labels[0]] != 'none':
                    total_num_ind += 1
                    total_num_sub += 1
                    if total_num_sub == 1:
                        data_final = data[i:(i+1)]
                        target_final = target[i:(i+1)]
                        pred_final = pred[i:(i+1)]
                        output_all_final = output_all[i:(i+1)]
                    else:
                        data_final = torch.cat((data_final, data[i:(i+1)]), dim = 0)
                        target_final = torch.cat((target_final, target[i:(i+1)]), dim = 0)
                        pred_final = torch.cat((pred_final, pred[i:(i+1)]), dim = 0)
                        output_all_final = torch.cat((output_all_final, output_all[i:(i+1)]), dim = 0)
                        
                    approx = torch.Tensor(explanation_all.local_pred_proba).unsqueeze(0)
                    approx_fixed = torch.Tensor(explanation.local_pred_proba).unsqueeze(0)
                    #approx = explanation_all.top_labels[np.argmax(explanation_all.local_pred_proba, axis = -1)]
                    #approx_fixed = explanation.top_labels[np.argmax(explanation.local_pred_proba, axis = -1)]
                    index0 = cuda(torch.LongTensor([[x[0] for x in explanation.local_exp[explanation.top_labels[0]]]]).unsqueeze(0), args.cuda)
            
                    if args.chunk_size > 1:
                        index = index_transfer(dataset = args.dataset,
                                                 idx = index0, 
                                                 filter_size = args.filter_size,
                                                 original_nrow = args.original_nrow,
                                                 original_ncol = args.original_ncol, 
                                                 is_cuda = args.cuda).output
                    else:
                        index = index0.squeeze(0)
                else:
                    j += 1

            elif 'imdb' in args.dataset:
            
                ## Calculate Gradient
                input = Variable(data[i:(i+1)])
                input_embeded = Variable(model.embedding_layer(input), requires_grad = False)

                explanation = args.explainer.explain_instance(input_embeded, 
                                         filter_size = args.filter_size, 
                                         classifier_fn = model.forward_sub, 
                                         top_labels = 2, 
                                         hide_color = 0, 
                                         num_features = k,
                                         num_samples = args.num_samples, 
                                         model_regressor = args.model_regressor,
                                         segments = args.segments,
                                         segments_data = args.segments_data)
                explanation_all = args.explainer_all.explain_instance(input_embeded, 
                                         filter_size = args.filter_size,
                                         classifier_fn = model.forward_sub, 
                                         top_labels = 2, 
                                         hide_color = 0, 
                                         #num_features = k,
                                         num_samples = args.num_samples, 
                                         model_regressor = args.model_regressor,
                                         segments = args.segments,
                                         segments_data = args.segments_data)

                if explanation.local_exp[explanation.top_labels[0]] != 'none':

                    total_num_ind += 1
                    total_num_sub += 1
                    if total_num_sub == 1:
                        data_final = data[0:1]
                        target_final = target[0:1]
                        pred_final = pred[0:1]
                        output_all_final = output_all[i:(i+1)]
                    else:
                        data_final = torch.cat((data_final, data[i:(i+1)]), dim = 0)
                        target_final = torch.cat((target_final, target[i:(i+1)]), dim = 0)
                        pred_final = torch.cat((pred_final, pred[i:(i+1)]), dim = 0)
                        output_all_final = torch.cat((output_all_final, output_all[i:(i+1)]), dim = 0)    
                    approx = torch.Tensor(explanation_all.local_pred_proba).unsqueeze(0)
                    approx_fixed = torch.Tensor(explanation.local_pred_proba).unsqueeze(0)
                    index0 = cuda(torch.LongTensor([x[0] for x in explanation.local_exp[explanation.top_labels[0]]]).unsqueeze(0), args.cuda)
                    
                    if args.chunk_size == args.original_ncol:
                        index = index_transfer(dataset = args.dataset,
                                           idx = index0,
                                           filter_size = args.filter_size,
                                           original_nrow = args.original_nrow,
                                           original_ncol = args.original_ncol,
                                           is_cuda = args.cuda).output

                    elif args.chunk_size > 1:
                        index = index_transfer(dataset = args.dataset,
                                                 idx = index0.unsqueeze(0), 
                                                 filter_size = args.filter_size,
                                                 original_nrow = args.original_nrow,
                                                 original_ncol = args.original_ncol, 
                                                 is_cuda = args.cuda).output.squeeze(0)
                    else:
                        index = index0

                else:
                    j += 1
                    #output_all[i] = output_all[i-1]
                    #pred[i] = pred[i-1]
                    #data[i] = data[i-1]
                    #if i == 0:
                    #    approx = torch.Tensor(explanation_all.local_pred_proba).unsqueeze(0)
                    #    approx_fixed = torch.Tensor(explanation.local_pred_proba).unsqueeze(0)
                    #    index0 = cuda(torch.LongTensor([range(k)]), args.cuda)

            if explanation.local_exp[explanation.top_labels[0]] != 'none':
                #if i == 0:
                if total_num_sub == 1:
                    approx_all = approx
                    approx_fixed_all = approx_fixed
                    index_all = index
                else:
                    approx_all = torch.cat((approx_all, approx), dim = 0)
                    approx_fixed_all = torch.cat((approx_fixed_all, approx_fixed), dim = 0)
                    index_all = torch.cat((index_all, index), dim = 0)

        ##define data_final
        if 'mnist' in args.dataset:

            data_size = data_final.size()
            binary_selected_all = idxtobool(index_all, [data_size[0], data_size[2] * data_size[3]], is_cuda)            
            data_zeropadded = torch.addcmul(torch.zeros(1), value=1, tensor1=binary_selected_all.view(data_size).type(torch.FloatTensor), tensor2=data_final.type(torch.FloatTensor), out=None)
        
        elif 'imdb' in args.dataset:
        
            data_size = data_final.size()
            binary_selected_all = idxtobool(index_all, [data_size[0], data_size[1]], is_cuda)            
            data_zeropadded = torch.addcmul(torch.zeros(1), value=1, tensor1=binary_selected_all.view(data_size).type(torch.FloatTensor), tensor2=data_final.type(torch.FloatTensor), out=None)
            data_zeropadded = data_zeropadded.type(torch.LongTensor)
        
        else:
        
            raise UnknownDatasetError()

        print('Batch {} Time spent is {}'.format(idx, time.time() - start)) 
        # Post-hoc Accuracy (zero-padded accuracy)
        output_zeropadded = model(cuda(data_zeropadded, is_cuda))             
        pred_zeropadded = output_zeropadded.max(1, keepdim=True)[1] # get the index of the max log-probability         
        correct_zeropadded += pred_zeropadded.eq(pred_final).sum().item()
   
        precision_macro_zeropadded += precision_score(pred_final, pred_zeropadded, average = 'macro')  
        precision_micro_zeropadded += precision_score(pred_final, pred_zeropadded, average = 'micro')  
        precision_weighted_zeropadded += precision_score(pred_final, pred_zeropadded, average = 'weighted')
        recall_macro_zeropadded += recall_score(pred_final, pred_zeropadded, average = 'macro')
        recall_micro_zeropadded += recall_score(pred_final, pred_zeropadded, average = 'micro')
        recall_weighted_zeropadded += recall_score(pred_final, pred_zeropadded, average = 'weighted')
        f1_macro_zeropadded += f1_score(pred_final, pred_zeropadded, average = 'macro')
        f1_micro_zeropadded += f1_score(pred_final, pred_zeropadded, average = 'micro')
        f1_weighted_zeropadded += f1_score(pred_final, pred_zeropadded, average = 'weighted')

        ## Variational Mutual Information
        vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1,
                                      tensor1 = torch.exp(output_all_final).type(torch.FloatTensor),
                                      tensor2 = output_zeropadded.type(torch.FloatTensor) -torch.sum(output_all_final, dim = -1).unsqueeze(-1).expand(output_zeropadded.size()).type(torch.FloatTensor),
                                      out=None), dim = -1)
        vmi = vmi.sum().item()
        vmi_zeropadded_sum += vmi

        ## Approximation Fidelity (prediction performance)
        pred = pred_final.type(torch.LongTensor)
        pred_approx = approx_all.topk(1, dim = -1)[1]
        #pred_approx = torch.Tensor(approx_all).type(torch.LongTensor).unsqueeze(-1)
        pred_approx_fixed = approx_fixed_all.topk(1, dim = -1)[1]
        #pred_approx_fixed = torch.Tensor(approx_fixed_all).type(torch.LongTensor).unsqueeze(-1)

        pred_approx_logit = F.softmax(torch.log(approx_all), dim=1)
        pred_approx_fixed_logit = F.softmax(torch.log(approx_fixed_all), dim = -1)
        correct_approx += pred_approx.eq(pred).sum().item()
        precision_macro_approx += precision_score(pred, pred_approx, average = 'macro')  
        precision_micro_approx += precision_score(pred, pred_approx, average = 'micro')  
        precision_weighted_approx += precision_score(pred, pred_approx, average = 'weighted')
        recall_macro_approx += recall_score(pred, pred_approx, average = 'macro')
        recall_micro_approx += recall_score(pred, pred_approx, average = 'micro')
        recall_weighted_approx += recall_score(pred, pred_approx, average = 'weighted')
        f1_macro_approx += f1_score(pred, pred_approx, average = 'macro')
        f1_micro_approx += f1_score(pred, pred_approx, average = 'micro')
        f1_weighted_approx += f1_score(pred, pred_approx, average = 'weighted')
        
        correct_approx_fixed += pred_approx_fixed.eq(pred).sum().item()
        precision_macro_approx_fixed += precision_score(pred, pred_approx_fixed, average = 'macro')  
        precision_micro_approx_fixed += precision_score(pred, pred_approx_fixed, average = 'micro')  
        precision_weighted_approx_fixed += precision_score(pred, pred_approx_fixed, average = 'weighted')
        recall_macro_approx_fixed += recall_score(pred, pred_approx_fixed, average = 'macro')
        recall_micro_approx_fixed += recall_score(pred, pred_approx_fixed, average = 'micro')
        recall_weighted_approx_fixed += recall_score(pred, pred_approx_fixed, average = 'weighted')
        f1_macro_approx_fixed += f1_score(pred, pred_approx_fixed, average = 'macro')
        f1_micro_approx_fixed += f1_score(pred, pred_approx_fixed, average = 'micro')
        f1_weighted_approx_fixed += f1_score(pred, pred_approx_fixed, average = 'weighted')

        vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1,
                                      tensor1 = torch.exp(output_all_final).type(torch.FloatTensor),
                                      tensor2 = pred_approx_logit.type(torch.FloatTensor) - torch.logsumexp(output_all_final, dim= 0).unsqueeze(0).expand(pred_approx_logit.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_all_final.size(0)).type(torch.FloatTensor)),
                                      out=None), dim = -1)
        vmi_fidel_sum += vmi.sum().item()
        vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1,
                                      tensor1 = torch.exp(output_all_final).type(torch.FloatTensor),
                                      tensor2 = pred_approx_fixed_logit.type(torch.FloatTensor) - torch.logsumexp(output_all_final, dim = 0).unsqueeze(0).expand(pred_approx_fixed_logit.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_all_final.size(0)).type(torch.FloatTensor)), out=None), dim = -1)
        vmi_fidel_fixed_sum += vmi.sum().item()  
#%%
        #print("SAVED!!!!")
        if idx in idx_list:

            # filename
            mname = re.sub(r'\.ckpt$', '', str(args.model_name))
            filename = 'figure_lime_' + args.dataset + '_model_' + mname + '_chunk' + str(args.chunk_size) + '_' + str(k) + '_idx' + str(idx) + '.png'
            filename = Path(args.out_dir).joinpath(filename)
            index_chunk = index_all
            
#            if args.chunk_size is not 1:
#                
#                index_chunk = index_transfer(dataset = args.dataset,
#                                             idx = index_chunk, 
#                                             filter_size = args.filter_size,
#                                             original_nrow = args.original_nrow,
#                                             original_ncol = args.original_ncol, 
#                                             is_cuda = args.cuda).output
            
            save_batch(dataset = args.dataset, 
                       batch = data_final, label = target_final, label_pred = pred_final.squeeze(-1), label_approx = pred_approx_fixed.squeeze(-1),
                       index = index_chunk, 
                       filename = filename, 
                       is_cuda = args.cuda,
                       word_idx = args.word_idx).output

        print('[summary] Batch {} Time spent is {}'.format(idx, time.time() - start))                   
#%%                               
    ## Post-hoc Accuracy (zero-padded accuracy)
    accuracy_zeropadded = correct_zeropadded/total_num_ind
    precision_macro_zeropadded = precision_macro_zeropadded/total_num
    precision_micro_zeropadded = precision_micro_zeropadded/total_num
    precision_weighted_zeropadded = precision_weighted_zeropadded/total_num
    recall_macro_zeropadded = recall_macro_zeropadded/total_num
    recall_micro_zeropadded = recall_micro_zeropadded/total_num
    recall_weighted_zeropadded = recall_weighted_zeropadded/total_num
    f1_macro_zeropadded = f1_macro_zeropadded/total_num
    f1_micro_zeropadded = f1_micro_zeropadded/total_num
    f1_weighted_zeropadded = f1_weighted_zeropadded/total_num

    ## VMI
    vmi_zeropadded = vmi_zeropadded_sum/total_num_ind
    vmi_fidel = vmi_fidel_sum / total_num_ind
    vmi_fidel_fixed = vmi_fidel_fixed_sum / total_num_ind
    
    ## Approximation Fidelity (prediction performance)
    accuracy_approx = correct_approx/total_num_ind
    precision_macro_approx = precision_macro_approx/total_num
    precision_micro_approx = precision_micro_approx/total_num
    precision_weighted_approx = precision_weighted_approx/total_num
    recall_macro_approx = recall_macro_approx/total_num
    recall_micro_approx = recall_micro_approx/total_num
    recall_weighted_approx = recall_weighted_approx/total_num
    f1_macro_approx = f1_macro_approx/total_num
    f1_micro_approx = f1_micro_approx/total_num
    f1_weighted_approx = f1_weighted_approx/total_num
    
    accuracy_approx_fixed = correct_approx_fixed/total_num_ind
    precision_macro_approx_fixed = precision_macro_approx_fixed/total_num
    precision_micro_approx_fixed = precision_micro_approx_fixed/total_num
    precision_weighted_approx_fixed = precision_weighted_approx_fixed/total_num
    recall_macro_approx_fixed = recall_macro_approx_fixed/total_num
    recall_micro_approx_fixed = recall_micro_approx_fixed/total_num
    recall_weighted_approx_fixed = recall_weighted_approx_fixed/total_num
    f1_macro_approx_fixed = f1_macro_approx_fixed/total_num
    f1_micro_approx_fixed = f1_micro_approx_fixed/total_num
    f1_weighted_approx_fixed = f1_weighted_approx/total_num
    
    ## Approximation Fidelity (prediction performance)

    print('\n\n[VAL RESULT]\n')
    #tab = pd.crosstab(y_class, prediction)
    #print(tab, end = "\n")                
    #print('IZY:{:.2f} IZX:{:.2f}'
    #        .format(izy_bound.item(), izx_bound.item()), end = '\n')
    print('acc_zeropadded:{:.4f} avg_acc:{:.4f} avg_acc_fixed:{:.4f}'.format(accuracy_zeropadded, accuracy_approx, accuracy_approx_fixed), end = '\n')    
    print('precision_macro_zeropadded:{:.4f} precision_macro_approx:{:.4f} precision_macro_approx_fixed:{:.4f}'
            .format(precision_macro_zeropadded, precision_macro_approx, precision_macro_approx_fixed), end = '\n')   
    print('precision_micro_zeropadded:{:.4f} precision_micro_approx:{:.4f} precision_micro_approx_fixed:{:.4f}'
            .format(precision_micro_zeropadded, precision_micro_approx, precision_micro_approx_fixed), end = '\n')   
    print('recall_macro_zeropadded:{:.4f} recall_macro_approx:{:.4f} recall_macro_approx_fixed:{:.4f}'
            .format(recall_macro_zeropadded, recall_macro_approx, recall_macro_approx_fixed), end = '\n')   
    print('recall_micro_zeropadded:{:.4f} recall_micro_approx:{:.4f} recall_micro_approx_fixed:{:.4f}'
            .format(recall_micro_zeropadded, recall_micro_approx, recall_micro_approx_fixed), end = '\n') 
    print('f1_macro_zeropadded:{:.4f} f1_macro_approx:{:.4f} f1_macro_approx_fixed:{:.4f}'
            .format(f1_macro_zeropadded, f1_macro_approx, f1_macro_approx_fixed), end = '\n')   
    print('f1_micro_zeropadded:{:.4f} f1_micro_approx:{:.4f} f1_micro_approx_fixed:{:.4f}'
            .format(f1_micro_zeropadded, f1_micro_approx_fixed, f1_micro_approx_fixed), end = '\n') 
    print('vmi:{:.4f} vmi_fixed:{:.4f} vmi_zeropadded:{:.4f}'.format(vmi_fidel, vmi_fidel_fixed, vmi_zeropadded), end = '\n')
    print()
    print('samples skipped{} total_num_ind{}'.format(j, total_num_ind))
    print("[END]")
    
#%%
#        if outfile:
#            
#            predictions = np.array(predictions)
#            predictions_idx = np.array(predictions_idx)
#            inds = predictions_idx.argsort()
#            sorted_predictions = predictions[inds]
#
#            output_name = model_name + '_pred_' + outmode + '.pt'
#            torch.save(sorted_predictions, Path(outfile_path).joinpath(output_name))                

if __name__ == '__main__':
    main()






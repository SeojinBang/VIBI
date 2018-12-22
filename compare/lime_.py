#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:12:32 2018

@author: seojin.bang
"""
import argparse
import os
import sys
import copy
import matplotlib.pyplot as plt
sys.path.append('../')
import numpy as np
from return_data import return_data
#from scipy import misc
#import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import str2bool, label2binary, cuda, idxtobool, UnknownDatasetError, index_transfer, save_batch
from pathlib import Path
from lib.image_utils import Segmentation
from torch.nn import functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression # Ridge
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')
#from torchvision.models import vgg16, vgg19
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
from lib.lime_library import LimeImageExplainer, ImageExplanation 
from lib.image_utils import Segmentation


#%%    
explainer = LimeImageExplainer(verbose = False)
#segmenter0 = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
segmenter = Segmentation(dataset = args.dataset, filter_size = (4, 4))
#%%
model_regressor = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='multinomial')#.fit(X, y)
model_regressor = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='ovr')#.fit(X, y)

#%%
model_regressor = LogisticRegression(class_weight = XX, XX, )
explanation = args.explainer.explain_instance(input, 
                         classifier_fn = model, 
                         top_labels = 1, 
                         hide_color = 0, 
                         #num_features = k,
                         num_samples = 10000, 
                         model_regressor = model_regressor,
                         segmentation_fn = args.segmenter)
                
#%%
image= input
classifier_fn = model
top_labels = 1
hide_color = 0
num_samples = 1000
segmentation_fn=segmenter
verbose = False

batch_size=10
distance_metric='cosine'
model_regressor=None
random_seed=None
labels = (1,)


#%%
 

#%%
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
    parser.add_argument('--cuda', default = True, type = str2bool, help = 'enable cuda')
    parser.add_argument('--out_dir', type=str, default='./result/lime/', help='Result directory path')
    parser.add_argument('--K', type = int, default = -1, help='dimension of encoding Z')
    
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
        args.explainer = LimeImageExplainer(verbose = False)
        args.segmenter = Segmentation(
                dataset = args.dataset,
                filter_size = args.filter_size,
                is_cuda = args.cuda)
    
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
        
    else:
    
        raise UnknownDatasetError()
            
#%%
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
                        
    #outmode = "TEST"

    #if outfile: 
    #    assert kargs['outmode'] in ['train', 'test', 'valid']
    #    outmode = kargs['outmode']
    
    #with torch.no_grad():
        
    #predictions = []
    #predictions_idx = []
#%%  
    for idx, batch in enumerate(test_loader):#(data, target, _, _)
#%%
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
        if 'imdb' in args.dataset: data_embeded = model.embedding_layer(data)
        output_all = model(data)
        test_loss += F.cross_entropy(output_all, target, reduction = 'sum').item()
        pred = output_all.max(1, keepdim=True)[1] # get the index of the max log-probability         
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_num += 1
        total_num_ind += data.size(0)
#%%            
        for i in range(data.size(0)):
            
            if 'mnist' in args.dataset:
            
                ## Calculate Gradient
                input = Variable(data[i:(i+1)], requires_grad = False) 
                output = model(input)
                output = torch.max(output)
                explanation = args.explainer.explain_instance(input, 
                                         classifier_fn = model, 
                                         top_labels = 1, 
                                         hide_color = 0, 
                                         #num_features = k,
                                         num_samples = 10000, 
                                         segmentation_fn = args.segmenter)
                
                grad_lime = explanation.local_exp
                
                #grad = (1, 1, 28, 28) tensor

                ## Select Variables
                grad_size = grad.size()
                if args.chunk_size > 1:
                    grad_chunk = F.avg_pool2d(torch.abs(grad), kernel_size = args.filter_size, stride = args.filter_size, padding = 0)
                    _, index_chunk = torch.abs(grad_chunk.view(grad_size[0], grad_size[1], -1)).topk(k, dim = -1)
                    index = index_transfer(dataset = args.dataset,
                                                 idx = index_chunk, 
                                                 filter_size = args.filter_size,
                                                 original_nrow = args.original_nrow,
                                                 original_ncol = args.original_ncol, 
                                                 is_cuda = args.cuda).output.unsqueeze(1)
                else:
                    grad_chunk = grad
                    _, index = torch.abs(grad_chunk.view(grad_size[0], grad_size[1], grad_size[2] * grad_size[3])).topk(k, dim = -1)

                ## Approximation
                grad_selected = grad.view(grad_size[0], grad_size[1], grad_size[2] * grad_size[3])[:,:,index[0][0]]
                data_selected = input.view(grad_size[0], grad_size[1], grad_size[2] * grad_size[3])[:,:,index[0][0]]
                
            elif 'imdb' in args.dataset:
            
                ## Calculate Gradient
                input = Variable(data[i:(i+1)])
                input_embeded = Variable(model.embedding_layer(input), requires_grad = False)
                
                output = model.forward_sub(input_embeded)
                output = torch.max(output)
                grad, = torch.autograd.grad(output, input_embeded, retain_graph = True)
                
                ## Select Variables
                grad_size = torch.Size([grad.size(0), args.max_num_sents, args.max_num_words])
                if args.chunk_size == args.max_num_words:
                    grad_chunk = torch.sum(torch.abs(grad.view(grad.size(0), args.max_num_sents, -1)), dim = -1)
                    _, index_chunk = grad_chunk.topk(k, dim = -1)
                    index = index_transfer(dataset = args.dataset,
                                                 idx = index_chunk, 
                                                 filter_size = args.filter_size,
                                                 original_nrow = args.original_nrow,
                                                 original_ncol = args.original_ncol, 
                                                 is_cuda = args.cuda).output#.unsqueeze(1)#[1,150]
                    
                elif args.chunk_size > 1:
                    grad_chunk = torch.sum(torch.abs(grad), dim = -1)
                    grad_chunk = grad_chunk.unsqueeze(-1).view(grad_size[0], grad_size[1], grad_size[2])
                    grad_chunk = F.avg_pool1d(grad_chunk, kernel_size = args.chunk_size, stride = 1, padding = 0)
                    
                    _, index_chunk = torch.abs(grad_chunk.view(grad_size[0], 1, -1)).topk(k, dim = -1)
                    index = index_transfer(dataset = args.dataset,#??
                                                 idx = index_chunk, 
                                                 filter_size = args.filter_size,
                                                 original_nrow = args.original_nrow,
                                                 original_ncol = args.original_ncol, 
                                                 is_cuda = args.cuda).output.squeeze(1)
                    
                else:
                    grad_chunk = torch.sum(torch.abs(grad), dim = -1)
                    _, index = torch.abs(grad_chunk.view(grad_size[0], grad_size[1] * grad_size[2])).topk(k, dim = -1)#[1,3]
                
                ## Approximation
                grad_selected = grad.view(grad_size[0], grad_size[1] * grad_size[2], -1)[:,index[0],:]
                data_selected = input.view(grad_size[0], grad_size[1] * grad_size[2], -1)[:,index[0],:]
            
            else:
            
                raise UnknownDatasetError()    

            if i == 0:
                grad_all = grad
                index_all = index
                grad_selected_all = grad_selected
                data_selected_all = data_selected
            else:
                grad_all = torch.cat((grad_all, grad), dim = 0) 
                index_all = torch.cat((index_all, index), dim = 0)
                grad_selected_all = torch.cat((grad_selected_all, grad_selected), dim = 0)
                data_selected_all = torch.cat((data_selected_all, data_selected), dim = 0)
                
        if 'mnist' in args.dataset:
        
            data_size = data.size()
            binary_selected_all = idxtobool(index_all, [data_size[0], data_size[1], data_size[2] * data_size[3]], is_cuda)            
            data_zeropadded = torch.addcmul(torch.zeros(1), value=1, tensor1=binary_selected_all.view(data_size).type(torch.FloatTensor), tensor2=data.type(torch.FloatTensor), out=None)
        
        elif 'imdb' in args.dataset:
        
            data_size = data.size()
            binary_selected_all = idxtobool(index_all, [data_size[0], data_size[1]], is_cuda)            
            data_zeropadded = torch.addcmul(torch.zeros(1), value=1, tensor1=binary_selected_all.view(data_size).type(torch.FloatTensor), tensor2=data.type(torch.FloatTensor), out=None)
            data_zeropadded = data_zeropadded.type(torch.LongTensor)
        
        else:
        
            raise UnknownDatasetError()
            
        # Post-hoc Accuracy (zero-padded accuracy)
        output_zeropadded = model(cuda(data_zeropadded, is_cuda))             
        pred_zeropadded = output_zeropadded.max(1, keepdim=True)[1] # get the index of the max log-probability         
        correct_zeropadded += pred_zeropadded.eq(pred).sum().item()
   
        precision_macro_zeropadded += precision_score(pred, pred_zeropadded, average = 'macro')  
        precision_micro_zeropadded += precision_score(pred, pred_zeropadded, average = 'micro')  
        precision_weighted_zeropadded += precision_score(pred, pred_zeropadded, average = 'weighted')
        recall_macro_zeropadded += recall_score(pred, pred_zeropadded, average = 'macro')
        recall_micro_zeropadded += recall_score(pred, pred_zeropadded, average = 'micro')
        recall_weighted_zeropadded += recall_score(pred, pred_zeropadded, average = 'weighted')
        f1_macro_zeropadded += f1_score(pred, pred_zeropadded, average = 'macro')
        f1_micro_zeropadded += f1_score(pred, pred_zeropadded, average = 'micro')
        f1_weighted_zeropadded += f1_score(pred, pred_zeropadded, average = 'weighted')

        ## Variational Mutual Information            
        vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                      tensor1 = torch.exp(output_all).type(torch.FloatTensor),
                                      tensor2 = output_zeropadded.type(torch.FloatTensor) - torch.sum(output_all, dim = -1).unsqueeze(-1).expand(output_zeropadded.size()).type(torch.FloatTensor),
                                      out=None), dim = -1)
        vmi = vmi.sum().item()
        vmi_zeropadded_sum += vmi
          
        ## Approximation Fidelity (prediction performance)
        for outidx in range(num_labels):

            for i in range(data.size(0)):

                if 'mnist' in args.dataset:
                
                    ## Calculate Gradient
                    input = Variable(data[i:(i+1), :, :, :], requires_grad = False) 
                    output = model(input)
                    output = output[0][outidx]
                    grad, = torch.autograd.grad(output, input, retain_graph = True)
                    
                    ## Select Variables
                    grad_size = grad.size()
                    if args.chunk_size > 1:
                        grad_chunk = F.avg_pool2d(torch.abs(grad), kernel_size = args.filter_size, stride = args.filter_size, padding = 0)
                        _, index_chunk = torch.abs(grad_chunk.view(grad_size[0], grad_size[1], -1)).topk(k, dim = -1)
                        index = index_transfer(dataset = args.dataset,
                                                     idx = index_chunk, 
                                                     filter_size = args.filter_size,
                                                     original_nrow = args.original_nrow,
                                                     original_ncol = args.original_ncol, 
                                                     is_cuda = args.cuda).output.unsqueeze(1)
                    else:
                        grad_chunk = grad
                        _, index = torch.abs(grad_chunk.view(grad_size[0], grad_size[1], grad_size[2] * grad_size[3])).topk(k, dim = -1)
                    
                    ## Approximation
                    grad_selected = grad.view(grad_size[0], grad_size[1], grad_size[2] * grad_size[3])[:,:,index[0][0]]
                    data_selected = input.view(grad_size[0], grad_size[1], grad_size[2] * grad_size[3])[:,:,index[0][0]]
                    
                elif 'imdb' in args.dataset:
                
                    ## Calculate Gradient
                    input = Variable(data[i:(i+1)])
                    input_embeded = Variable(model.embedding_layer(input), requires_grad = False)
                    output = model.forward_sub(input_embeded)
                    output = output[0][outidx]
                    grad, = torch.autograd.grad(output, input_embeded, retain_graph = True)
                    
                    ## Select Variables
                    grad_size = torch.Size([grad.size(0), args.max_num_sents, args.max_num_words])
                    if args.chunk_size == args.max_num_words:
                        grad_chunk = torch.sum(torch.abs(grad.view(grad.size(0), args.max_num_sents, -1)), dim = -1)
                        _, index_chunk = grad_chunk.topk(k, dim = -1)
                        index = index_transfer(dataset = args.dataset,
                                                     idx = index_chunk, 
                                                     filter_size = args.filter_size,
                                                     original_nrow = args.original_nrow,
                                                     original_ncol = args.original_ncol, 
                                                     is_cuda = args.cuda).output
                        
                    elif args.chunk_size > 1:
                        grad_chunk = torch.sum(torch.abs(grad), dim = -1)
                        grad_chunk = grad_chunk.unsqueeze(-1).view(grad_size[0], grad_size[1], grad_size[2])
                        grad_chunk = F.avg_pool1d(grad_chunk, kernel_size = args.chunk_size, stride = 1, padding = 0)
                        _, index_chunk = torch.abs(grad_chunk.view(grad_size[0], 1, -1)).topk(k, dim = -1)
                        index = index_transfer(dataset = args.dataset, 
                                                     idx = index_chunk, 
                                                     filter_size = args.filter_size,
                                                     original_nrow = args.original_nrow,
                                                     original_ncol = args.original_ncol, 
                                                     is_cuda = args.cuda).output.squeeze(1)
                    
                    else:
                        grad_word = torch.sum(torch.abs(grad), dim = -1)
                        _, index = torch.abs(grad_word).topk(k, dim = -1)
                        
                    ## Approximation
                    grad_selected = grad[:,index[0],:]
                    data_selected = input_embeded[:,index[0],:]
                
                else:
                
                    raise UnknownDatasetError()    

                if i == 0:
                    grad_all = grad
                    index_all = index
                    grad_selected_all = grad_selected
                    data_selected_all = data_selected
                else:
                    grad_all = torch.cat((grad_all, grad), dim = 0) 
                    index_all = torch.cat((index_all, index), dim = 0)
                    grad_selected_all = torch.cat((grad_selected_all, grad_selected), dim = 0)
                    data_selected_all = torch.cat((data_selected_all, data_selected), dim = 0)
            
            if 'mnist' in args.dataset:
            
                approx = torch.addcmul(torch.zeros(1), value = 1, tensor1 = grad_all.view(data_size[0], data_size[1], data_size[2] * data_size[3]).type(torch.FloatTensor), tensor2 = data.view(data_size[0], data_size[1], data_size[2] * data_size[3]).type(torch.FloatTensor), out=None)
                approx = torch.exp(torch.sum(approx, dim = -1))##squeeze(-1)
                approx_fixed = torch.addcmul(torch.zeros(1), value = 1, tensor1 = grad_selected_all.type(torch.FloatTensor) , tensor2 = data_selected_all.type(torch.FloatTensor), out=None)
                approx_fixed = torch.exp(torch.sum(approx_fixed, dim = -1))#.squeeze(-1)
                
            elif 'imdb' in args.dataset:
                
                approx = torch.addcmul(torch.zeros(1), value = 1, tensor1 = grad_all.type(torch.FloatTensor), tensor2 = data_embeded.type(torch.FloatTensor), out=None)
                approx = torch.exp(torch.sum(torch.sum(approx, dim = -1), dim = -1)).unsqueeze(-1)
                approx_fixed = torch.addcmul(torch.zeros(1), value = 1, tensor1 = grad_selected_all.type(torch.FloatTensor) , tensor2 = data_selected_all.type(torch.FloatTensor), out=None)
                approx_fixed = torch.exp(torch.sum(torch.sum(approx_fixed, dim = -1), dim = -1)).unsqueeze(-1)
            
            else:
                
                raise UnknownDatasetError()   
            
            if outidx == 0:
                approx_all = approx
                approx_fixed_all = approx_fixed
            else:
                approx_all = torch.cat((approx_all, approx), dim = 1)
                approx_fixed_all = torch.cat((approx_fixed_all, approx_fixed), dim = 1)

        pred = pred.type(torch.LongTensor)
        pred_approx = approx_all.topk(1, dim = -1)[1]
        pred_approx = pred_approx.type(torch.LongTensor)
        pred_approx_fixed = approx_fixed_all.topk(1, dim = -1)[1]
        pred_approx_fixed = pred_approx_fixed.type(torch.LongTensor)
              
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
##%%
#        #if (idx == 0 or idx == 200): ## figure
#        if idx in idx_list:
#
#            filename = 'figure_saliency_' + args.dataset + '_' + str(k) + '_idx' + str(idx) + '.png'
#            filename = Path(args.out_dir).joinpath(filename)
#    
#            #img = copy.deepcopy(grad_all)    
#            img = copy.deepcopy(data)
#            n_img = img.size(0)
#            n_col = 8
#            n_row = n_img // n_col + 1
#    
#            fig = plt.figure(figsize=(n_col * 1.5, n_row * 1.5)) 
#    
#            for i in range(n_img):
#    
#                plt.subplot(n_row, n_col, 1 + i)
#                plt.axis('off')
#                # original image
#                img0 = img[i].squeeze(0)#.numpy()
#                plt.imshow(img0, cmap = 'autumn_r')
#                # chunk selected
#                img2 = img[i].view(-1)#.numpy()
#                img2[index_all[i]] = cuda(torch.tensor(float('nan')), is_cuda)
#                img2 = img2.view(img0.size())#.numpy()
#                plt.title('BB {}, Apx {}'.format(pred[i].item(), pred[i].item()))
#                plt.imshow(img2, cmap = 'gray')
#    
#            fig.subplots_adjust(wspace = 0.05, hspace = 0.35)       
#            fig.savefig(filename)
#            
#            ## Save predictions
#            #predictions.extend(pred.data.squeeze(-1).cpu().tolist())
#            #predictions_idx.extend(idx.cpu().tolist())

        #print("SAVED!!!!")
        if idx in idx_list:

            # filename
            filename = 'figure_saliency_' + args.dataset + '_chunk' + str(args.chunk_size) + '_' + str(k) + '_idx' + str(idx) + '.png'
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
                       batch = data, label = target, label_pred = pred.squeeze(-1), label_approx = pred_approx_fixed.squeeze(-1),
                       index = index_chunk, 
                       filename = filename, 
                       is_cuda = args.cuda,
                       word_idx = args.word_idx).output
#%%                                 
    ## Post-hoc Accuracy (zero-padded accuracy)
    vmi_zeropadded = vmi_zeropadded_sum/total_num_ind
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
    print('acc_zeropadded:{:.4f} avg_acc:{:.4f} avg_acc_fixed:{:.4f}'
            .format(accuracy_zeropadded, accuracy_approx, accuracy_approx_fixed), end = '\n')
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
    print('vmi:{:.4f}'.format(vmi_zeropadded), end = '\n')
    print()
    
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






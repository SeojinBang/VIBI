import os
import sys
#os.chdir('~/Users/seojin.bang/OneDrive\ -\ Petuum\,\ Inc/VIB-pytorch')dd
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
#from torch.utils.data import DataLoader
#from torchvision import transforms
from tensorboardX import SummaryWriter
from utils import cuda, Weight_EMA_Update, label2binary, save_batch, index_transfer, timeSince, UnknownDatasetError, idxtobool
from return_data import return_data
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
#import pandas as pd
#%%
class Solver(object):

    def __init__(self, args):
        
        self.args = args
        self.dataset = args.dataset
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr # learning rate
        self.beta = args.beta
        self.cuda = args.cuda
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.num_avg = args.num_avg
        self.global_iter = 0
        self.global_epoch = 0
        self.env_name = os.path.splitext(args.checkpoint_name)[0] if args.env_name is 'main' else args.env_name

        # Dataset
        self.args.root = os.path.join(self.args.dataset, self.args.data_dir)
        self.args.load_pred = True
        self.data_loader = return_data(args = self.args)
        self.args.word_idx = None
        
        if 'mnist' in self.dataset:

            for idx, (x,_,_,_) in enumerate(self.data_loader['train']):
                xsize = x.size()       
                if idx == 0: break

            self.d = torch.tensor(xsize[1:]).prod()
            self.x_type = self.data_loader['x_type']
            self.y_type = self.data_loader['y_type']
        
            sys.path.append("./" + self.dataset)

            self.original_ncol = 28
            self.original_nrow = 28
            self.args.chunk_size = self.args.chunk_size if self.args.chunk_size > 0 else 2
            self.chunk_size = self.args.chunk_size
            assert np.remainder(self.original_nrow, self.chunk_size) == 0
            self.filter_size = (self.chunk_size, self.chunk_size)
            
            ## load black box model
            from mnist.original import Net
            self.black_box = Net().to(self.device) 
        
        elif 'imdb' in self.dataset:

            for idx, batch in enumerate(self.data_loader['train']):
                xsize = batch.text.size()       
                if idx == 0: break
        
            self.d = torch.tensor(xsize[1:]).prod()
            self.x_type = self.data_loader['x_type']
            self.y_type = self.data_loader['y_type']
            
            sys.path.append("./" + self.dataset)

            self.args.word_idx = self.data_loader['word_idx']
            self.args.max_total_num_words = self.data_loader['max_total_num_words']
            self.args.embedding_dim = self.data_loader['embedding_dim']
            self.args.max_num_words = self.data_loader['max_num_words'] #100
            self.args.max_num_sents = self.data_loader['max_num_sents'] #15   
            self.original_ncol = self.args.max_num_words
            self.original_nrow = self.args.max_num_sents
            self.args.chunk_size = self.args.chunk_size if self.args.chunk_size > 0 else 4
            self.chunk_size = self.args.chunk_size
            if self.chunk_size > self.original_ncol: self.chunk_size = self.original_ncol
            self.filter_size = (1, self.chunk_size)

            ## load black box model
            
            from imdb.original import Net
            self.args.model_type = args.model_name.split('_')[-1].split('.')[0]
            self.black_box = Net(args = self.args).to(self.device)
                
        else:
            
            raise UnknownDatasetError()

    #%%
        # Black box
        self.args.model_dir = args.dataset + '/models'
        model_name = Path(self.args.model_dir).joinpath(self.args.model_name)
        self.black_box.load_state_dict(torch.load(model_name, map_location='cpu'))
    #%%
        if self.cuda:
            self.black_box.cuda()
            
        if torch.cuda.device_count() is 0:
            self.black_box.eval()
            
        elif 'lstm' in self.args.explainer_type:
            self.black_box.train()
                
        elif 'rnn' in self.args.explainer_type:
            self.black_box.train()
            
        elif 'gru' in self.args.explainer_type:
            self.black_box.train()
            
        else:
            self.black_box.eval() 
#%%  
        from explainer import Explainer, prior
        self.prior = prior

        # Network
        self.net = cuda(Explainer(args = self.args), self.args.cuda)
        self.net.weight_init()
        self.net_ema = Weight_EMA_Update(cuda(Explainer(args = self.args), self.args.cuda), self.net.state_dict(), decay = 0.999)
        
        # Optimizer
        self.optim = optim.Adam(self.net.parameters(),lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

        # Checkpoint
        self.checkpoint_dir = Path(args.dataset).joinpath(args.checkpoint_dir, args.env_name)
        if not self.checkpoint_dir.exists() : self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = Path(args.dataset).joinpath(args.checkpoint_dir, 'sample')
        if not self.image_dir.exists() : self.image_dir.mkdir(parents=True, exist_ok=True)
        self.load_checkpoint = args.load_checkpoint
        if self.load_checkpoint != '' : self.load_checkpoint(self.load_checkpoint)
        self.checkpoint_name = args.checkpoint_name

        # History
        self.history = dict()
        self.history['info_loss'] = 0.
        self.history['class_loss'] = 0.
        self.history['total_loss'] = 0.
        self.history['epoch'] = 0
        self.history['iter'] = 0
        
        self.history['avg_acc'] = 0.
        self.history['avg_acc_fixed'] = 0.
#        self.history['avg_auc_macro'] = 0.
#        self.history['avg_auc_micro'] = 0.
#        self.history['avg_auc_weighted'] = 0.
        self.history['avg_precision_macro'] = 0.
        self.history['avg_precision_micro'] = 0.
        self.history['avg_precision_fixed_macro'] = 0.
        self.history['avg_precision_fixed_micro'] = 0.
        #self.history['avg_precision_weighted'] = 0.
        self.history['avg_recall_macro'] = 0.
        self.history['avg_recall_micro'] = 0.
        self.history['avg_recall_fixed_macro'] = 0.
        self.history['avg_recall_fixed_micro'] = 0.
        #self.history['avg_recall_weighted'] = 0. 
        self.history['avg_f1_macro'] = 0.
        self.history['avg_f1_micro'] = 0.
        self.history['avg_f1_fixed_macro'] = 0.
        self.history['avg_f1_fixed_micro'] = 0.
        #self.history['avg_f1_weighted'] = 0.
 
        self.history['acc_zeropadded'] = 0.
        self.history['precision_macro_zeropadded'] = 0.
        self.history['precision_micro_zeropadded'] = 0.
#        self.history['precision_weighted_zeropadded'] = 0.
        self.history['recall_macro_zeropadded'] = 0.
        self.history['recall_micro_zeropadded'] = 0.
#        self.history['recall_weighted_zeropadded'] = 0. 
        self.history['f1_macro_zeropadded'] = 0.
        self.history['f1_micro_zeropadded'] = 0.
#        self.history['f1_weighted_zeropadded'] = 0.  
        self.history['vmi_zeropadded'] = 0.
        self.history['avg_vmi'] = 0.       
        self.history['avg_vmi_fixed'] = 0.

        # Tensorboard
        self.tensorboard = args.tensorboard
        if self.tensorboard :
            self.summary_dir = Path(args.dataset).joinpath(args.summary_dir, self.env_name)
            if not self.summary_dir.exists() : self.summary_dir.mkdir(parents = True, exist_ok = True)
            self.tf = SummaryWriter(log_dir = str(self.summary_dir))
            self.tf.add_text(tag='argument',text_string = str(args), global_step = self.global_epoch)
#%%
    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
            self.net_ema.model.train()
        
        elif mode == 'eval':     
            self.net.eval()
            self.net_ema.model.eval()
            
        else : raise('mode error. It should be either train or eval')
#%%
    def train(self):
#%%
        self.set_mode('train')
        #%%
        self.class_criterion = nn.CrossEntropyLoss(reduction = 'sum')
        self.info_criterion = nn.KLDivLoss(reduction = 'sum')
#%%     
        start = time.time()   
        for e in range(self.epoch) :
            #if e > 1: break
            self.global_epoch += 1
#%%
            for idx, batch in enumerate(self.data_loader['train']):
                
                if 'mnist' in self.dataset:

                    x_raw = batch[0]
                    y_raw = batch[2]
                    
                elif 'imdb' in self.dataset:
                    
                    x_raw = batch.text
                    y_raw = batch.label_pred.view(-1)
                
                else:
            
                    raise UnknownDatasetError()
      
                self.global_iter += 1
                
                x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)
#%%
                #raise ValueError('num_sample should be a positive integer') 
                logit, log_p_i, Z_hat, logit_fixed = self.net(x)

                ## prior distribution
                p_i_prior = cuda(self.prior(var_size = log_p_i.size()), self.args.cuda)

                ## define loss
                y_class = y if len(y.size()) == 1 else torch.argmax(y, dim = -1)
#                y_binary = label2binary(y_class, classes = range(logit.size(-1)))
#%%
                class_loss = self.class_criterion(logit, y_class).div(math.log(2)) / self.batch_size
                info_loss = self.args.K * self.info_criterion(log_p_i, p_i_prior) / self.batch_size
                total_loss = class_loss + self.beta * info_loss
                
                izy_bound = math.log(10,2) - class_loss
                izx_bound = info_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                self.net_ema.update(self.net.state_dict())

                prediction = torch.argmax(logit, dim = -1)
                accuracy = torch.eq(prediction, y_class).float().mean() 
                prediction_fixed = torch.argmax(logit_fixed, dim = -1)
                accuracy_fixed = torch.eq(prediction_fixed, y_class).float().mean()
#                auc_macro = roc_auc_score(y_binary, logit.detach().numpy(), average = 'macro')
#                auc_micro = roc_auc_score(y_binary, logit.detach().numpy(), average = 'micro')
                #auc_weighted = roc_auc_score(y_binary, logit.detach().numpy(), average = 'weighted')    
                precision_macro = precision_score(y_class, prediction, average = 'macro')  
                precision_micro = precision_score(y_class, prediction, average = 'micro')  
                precision_fixed_macro = precision_score(y_class, prediction_fixed, average = 'macro') 
                precision_fixed_micro = precision_score(y_class, prediction_fixed, average = 'micro')  
                #precision_weighted = precision_score(y_class, prediction, average = 'weighted')
                recall_macro = recall_score(y_class, prediction, average = 'macro')
                recall_micro = recall_score(y_class, prediction, average = 'micro')
                recall_fixed_macro = recall_score(y_class, prediction_fixed, average = 'macro')
                recall_fixed_micro = recall_score(y_class, prediction_fixed, average = 'micro')        
                #recall_weighted = recall_score(y_class, prediction, average = 'weighted')
                f1_macro = f1_score(y_class, prediction, average = 'macro')
                f1_micro = f1_score(y_class, prediction, average = 'micro')
                f1_fixed_macro = f1_score(y_class, prediction_fixed, average = 'macro')
                f1_fixed_micro = f1_score(y_class, prediction_fixed, average = 'micro')
                #f1_weighted = f1_score(y_class, prediction, average = 'weighted')

                # selected chunk index
                _, index_chunk = log_p_i.unsqueeze(1).topk(self.args.K, dim = -1)

                if self.chunk_size is not 1:
                    
                    index_chunk = index_transfer(dataset = self.dataset,
                                                 idx = index_chunk, 
                                                 filter_size = self.filter_size,
                                                 original_nrow = self.original_nrow,
                                                 original_ncol = self.original_ncol, 
                                                 is_cuda = self.cuda).output
#%%                                                         
                if 'mnist' in self.dataset:
                
                    data_size = x_raw.size()
                    binary_selected_all = idxtobool(index_chunk.view(data_size[0], data_size[1], -1), [data_size[0], data_size[1], data_size[2] * data_size[3]], self.cuda)            
                    data_zeropadded = torch.addcmul(torch.zeros(1), value = 1, 
                                                    tensor1=binary_selected_all.view(data_size).type(torch.FloatTensor), tensor2=x_raw.type(torch.FloatTensor), out=None)
                    data_zeropadded = data_zeropadded.type(self.x_type)
                    #data_zeropadded[data_zeropadded == 0] = -1
                    
                elif 'imdb' in self.dataset:
                
                    data_size = x_raw.size()
                    binary_selected_all = idxtobool(index_chunk.view(data_size[0], -1), [data_size[0], data_size[1]], self.cuda)            
                    data_zeropadded = torch.addcmul(torch.zeros(1), value=1, tensor1=binary_selected_all.view(data_size).type(torch.FloatTensor), tensor2=x_raw.type(torch.FloatTensor), out=None)
                    data_zeropadded = data_zeropadded.type(self.x_type)
                    data_zeropadded[data_zeropadded == 0] = 1
                
                else:
                
                    raise UnknownDatasetError()
#%%            
                # Post-hoc Accuracy (zero-padded accuracy)
                output_original = self.black_box(x)
                output_zeropadded = self.black_box(data_zeropadded)                
                pred_zeropadded = F.softmax(output_zeropadded, dim=1).max(1)[1]
                #pred_zeropadded = output_zeropadded.max(1, keepdim=True)[1] 
                accuracy_zeropadded = torch.eq(pred_zeropadded, y_class).float().mean() 
                print('train zeropadded', pred_zeropadded)
                print('train true', y_class)
           
                precision_macro_zeropadded = precision_score(y_class, pred_zeropadded, average = 'macro')  
                precision_micro_zeropadded = precision_score(y_class, pred_zeropadded, average = 'micro')  
                #precision_weighted_zeropadded = precision_score(y_class, pred_zeropadded, average = 'weighted')
                recall_macro_zeropadded = recall_score(y_class, pred_zeropadded, average = 'macro')
                recall_micro_zeropadded = recall_score(y_class, pred_zeropadded, average = 'micro')
                #recall_weighted_zeropadded = recall_score(y_class, pred_zeropadded, average = 'weighted')
                f1_macro_zeropadded = f1_score(y_class, pred_zeropadded, average = 'macro')
                f1_micro_zeropadded = f1_score(y_class, pred_zeropadded, average = 'micro')
                #f1_weighted_zeropadded = f1_score(y_class, pred_zeropadded, average = 'weighted')
        
                ## Variational Mutual Information            
                vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                              tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                              tensor2 = output_zeropadded.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(output_zeropadded.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                              out=None), dim = -1)
                vmi_zeropadded = vmi.mean()
                vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                              tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                              tensor2 = logit.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(logit.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                              out=None), dim = -1)
                vmi_fidel = vmi.mean()

                vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                              tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                              tensor2 = logit_fixed.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(logit_fixed.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                              out=None), dim = -1)
                vmi_fidel_fixed = vmi.mean()
                
                if self.num_avg != 0 :
                    avg_soft_logit, avg_log_p_i, _, avg_soft_logit_fixed = self.net(x, self.num_avg)
                    avg_prediction = avg_soft_logit.max(1)[1]
                    avg_accuracy = torch.eq(avg_prediction,y_class).float().mean()
                    avg_prediction_fixed  = avg_soft_logit_fixed.max(1)[1]
                    avg_accuracy_fixed  = torch.eq(avg_prediction_fixed,y_class).float().mean()                  
#                    avg_auc_macro = roc_auc_score(y_binary, avg_soft_logit.detach().numpy(), average = 'macro')
#                    avg_auc_micro = roc_auc_score(y_binary, avg_soft_logit.detach().numpy(), average = 'micro')
                    #avg_auc_weighted = roc_auc_score(y_binary, avg_soft_logit.detach().numpy(), average = 'weighted') 
                    avg_precision_macro = precision_score(y_class, avg_prediction, average = 'macro')  
                    avg_precision_micro = precision_score(y_class, avg_prediction, average = 'micro')  
                    avg_precision_fixed_macro = precision_score(y_class, avg_prediction_fixed, average = 'macro')  
                    avg_precision_fixed_micro = precision_score(y_class, avg_prediction_fixed, average = 'micro') 
                    #avg_precision_weighted = precision_score(y_class, avg_prediction, average = 'weighted')
                    avg_recall_macro = recall_score(y_class, avg_prediction, average = 'macro')
                    avg_recall_micro = recall_score(y_class, avg_prediction, average = 'micro')
                    avg_recall_fixed_macro = recall_score(y_class, avg_prediction_fixed, average = 'macro')
                    avg_recall_fixed_micro = recall_score(y_class, avg_prediction_fixed, average = 'micro')
                    #avg_recall_weighted = recall_score(y_class, avg_prediction, average = 'weighted')
                    avg_f1_macro = f1_score(y_class, avg_prediction, average = 'macro')
                    avg_f1_micro = f1_score(y_class, avg_prediction, average = 'micro')
                    avg_f1_fixed_macro = f1_score(y_class, avg_prediction_fixed, average = 'macro')
                    avg_f1_fixed_micro = f1_score(y_class, avg_prediction_fixed, average = 'micro')
                    #avg_f1_weighted = f1_score(y_class, avg_prediction, average = 'weighted')  
            
                    ## Variational Mutual Information            
                    vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                                  tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                                  tensor2 = avg_soft_logit.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(avg_soft_logit.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                                  out=None), dim = -1)
                    avg_vmi_fidel = vmi.mean()
    
                    vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                                  tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                                  tensor2 = avg_soft_logit_fixed.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(avg_soft_logit_fixed.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                                  out=None), dim = -1)
                    avg_vmi_fidel_fixed = vmi.mean()
                    
                else : 
#                    avg_accuracy = Variable(cuda(torch.zeros(accuracy.size()), self.args.cuda))
#                    #avg_auc_macro = Variable(cuda(torch.zeros(auc_macro.size()), self.args.cuda))
#                    #avg_auc_micro = Variable(cuda(torch.zeros(auc_micro.size()), self.args.cuda))
#                    #avg_auc_weighted = Variable(cuda(torch.zeros(auc_weighted.size()), self.args.cuda))
#                    avg_precision_macro = Variable(cuda(torch.zeros(precision_macro.size()), self.args.cuda))
#                    avg_precision_micro = Variable(cuda(torch.zeros(precision_micro.size()), self.args.cuda))
#                    #avg_precision_weighted = Variable(cuda(torch.zeros(precision_weighted.size()), self.args.cuda))
#                    avg_recall_macro = Variable(cuda(torch.zeros(recall_macro.size()), self.args.cuda))
#                    avg_recall_micro = Variable(cuda(torch.zeros(recall_micro.size()), self.args.cuda))
#                    #avg_recall_weighted = Variable(cuda(torch.zeros(recall_weighted.size()), self.args.cuda))
#                    avg_f1_macro = Variable(cuda(torch.zeros(f1_macro.size()), self.args.cuda))
#                    avg_f1_micro = Variable(cuda(torch.zeros(f1_micro.size()), self.args.cuda))
#                    #avg_f1_weighted = Variable(cuda(torch.zeros(f1_weighted.size()), self.args.cuda))
                    
                    avg_accuracy = accuracy
                    avg_accuracy_fixed = accuracy_fixed
#                    avg_auc_macro = auc_macro
#                    avg_auc_micro = auc_micro
                    #avg_auc_weighted = auc_weighted
                    avg_precision_macro = precision_macro
                    avg_precision_micro = precision_micro
                    avg_precision_fixed_macro = precision_fixed_macro
                    avg_precision_fixed_micro = precision_fixed_micro
                    #avg_precision_weighted = precision_weighted
                    avg_recall_macro = recall_macro
                    avg_recall_micro = recall_micro
                    avg_recall_fixed_macro = recall_fixed_macro
                    avg_recall_fixed_micro = recall_fixed_micro
                    #avg_recall_weighted = recall_weighted
                    avg_f1_macro = f1_macro
                    avg_f1_micro = f1_micro
                    avg_f1_fixed_macro = f1_fixed_macro
                    avg_f1_fixed_micro = f1_fixed_micro
                    #avg_f1_weighted = f1_weighted

                    avg_vmi_fidel = vmi_fidel
                    avg_vmi_fidel_fixed = vmi_fidel_fixed
                    
                if self.global_iter % 1000 == 0 :
                #     print('i:{} IZY:{:.2f} IZX:{:.2f}'.format(idx+1, izy_bound.item(), izx_bound.item()), end=' ')
                #     print('acc:{:.4f} avg_acc:{:.4f}'.format(accuracy.item(), avg_accuracy.item()), end=' ')
                #     print('err:{:.4f} avg_err:{:.4f}'.format(1-accuracy.item(), 1-avg_accuracy.item()))

                    print('\n\n[TRAINING RESULT]\n')
                    #print("logit", logit)
                    #print("y_class", y_class)
                    #print("prediction", prediction)
                    #tab = pd.crosstab(y_class, prediction)
                    #print(tab, end = "\n")
                    print('epoch {}'.format(self.global_epoch), end = "\n")
                    print('global iter {}'.format(self.global_iter), end = "\n")
                    print('i:{} IZY:{:.2f} IZX:{:.2f}'
                            .format(idx+1, izy_bound.item(), izx_bound.item()), end = '\n')
                    print('acc:{:.4f} avg_acc:{:.4f}'
                            .format(accuracy.item(), avg_accuracy.item()), end = '\n')
                    print('acc_fixed:{:.4f} avg_acc_fixed:{:.4f}'
                            .format(accuracy_fixed.item(), avg_accuracy_fixed.item()), end='\n')
                    print('vmi:{:.4f} avg_vmi:{:.4f}'
                            .format(vmi_fidel.item(), avg_vmi_fidel.item()), end = '\n')
                    print('vmi_fixed:{:.4f} avg_vmi_fixed:{:.4f}'
                            .format(vmi_fidel_fixed.item(), avg_vmi_fidel_fixed.item()), end = '\n')
                    print('acc_zeropadded:{:.4f} vmi_zeropadded:{:.4f}'
                            .format(accuracy_zeropadded.item(), vmi_zeropadded.item()), end = '\n')
##                    print('auc_macro:{:.4f} avg_auc_macro:{:.4f}'
##                            .format(auc_macro.item(), avg_auc_macro.item()), end = '\n')   
##                    print('auc_micro:{:.4f} avg_auc_micro:{:.4f}'
##                            .format(auc_micro.item(), avg_auc_micro.item()), end = '\n')     
#                    print('precision_macro:{:.4f} avg_precision_macro:{:.4f}'
#                            .format(precision_macro.item(), avg_precision_macro.item()), end='\n')   
#                    print('precision_micro:{:.4f} avg_precision_micro:{:.4f}'
#                            .format(precision_micro.item(), avg_precision_micro.item()), end='\n')   
#                    print('recall_macro:{:.4f} avg_recall_macro:{:.4f}'
#                            .format(recall_macro.item(), avg_recall_macro.item()), end='\n')   
#                    print('recall_micro:{:.4f} avg_recall_micro:{:.4f}'
#                            .format(recall_micro.item(), avg_recall_micro.item()), end='\n') 
#                    print('f1_macro:{:.4f} avg_f1_macro:{:.4f}'
#                            .format(f1_macro.item(), avg_f1_macro.item()), end='\n')   
#                    print('f1_micro:{:.4f} avg_f1_micro:{:.4f}'
#                            .format(f1_micro.item(), avg_f1_micro.item()), end='\n')                          
##                    print('err:{:.4f} avg_err:{:.4f}'
##                            .format(1-accuracy.item(), 1-avg_accuracy.item()))
#                    print('precision_macro_zeropadded:{:.4f}'
#                            .format(precision_macro_zeropadded.item()), end = '\n')
#                    print('precision_micro_zeropadded:{:.4f}'
#                            .format(precision_micro_zeropadded.item()), end = '\n')
#                    print('recall_macro_zeropadded:{:.4f}'
#                            .format(recall_macro_zeropadded.item()), end = '\n')   
#                    print('recall_micro_zeropadded:{:.4f}'
#                            .format(recall_micro_zeropadded.item()), end = '\n') 
#                    print('f1_macro_zeropadded:{:.4f}'
#                            .format(f1_macro_zeropadded.item()), end = '\n')   
#                    print('f1_micro_zeropadded:{:.4f}'
#                            .format(f1_micro_zeropadded.item()), end = '\n') 
            
                if self.global_iter % 10 == 0 :
                    if self.tensorboard :
                        self.tf.add_scalars(main_tag='performance/accuracy',
                                            tag_scalar_dict={
                                                'train_one-shot':accuracy.item(),
                                                'train_multi-shot':avg_accuracy.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/accuracy_fixed',
                                            tag_scalar_dict={
                                                'train_one-shot':accuracy_fixed.item(),
                                                'train_multi-shot':avg_accuracy_fixed.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/vmi',
                                            tag_scalar_dict={
                                                'train_one-shot':vmi_fidel.item(),
                                                'train_multi-shot':avg_vmi_fidel.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/vmi_fixed',
                                            tag_scalar_dict={
                                                'train_one-shot':vmi_fidel_fixed.item(),
                                                'train_multi-shot':avg_vmi_fidel_fixed.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/accuracy_zeropadded',
                                            tag_scalar_dict={
                                                'train_one-shot':accuracy_zeropadded.item()#,
                                                #'train_multi-shot':avg_accuracy_zeropadded.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/vmi_zeropadded',
                                            tag_scalar_dict={
                                                'train_one-shot':vmi_zeropadded.item()#,
                                                #'train_multi-shot':avg_accuracy_zeropadded.item()
                                                },
                                            global_step=self.global_iter)
#                        self.tf.add_scalars(main_tag='performance/error',
#                                            tag_scalar_dict={
#                                                'train_one-shot':1-accuracy.item(),
#                                                'train_multi-shot':1-avg_accuracy.item()
#                                                },
#                                            global_step=self.global_iter)
#                        self.tf.add_scalars(main_tag='performance/auc_macro',
#                                            tag_scalar_dict={
#                                                'train_one-shot':auc_macro.item(),
#                                                'train_multi-shot':avg_auc_macro.item()},
#                                            global_step=self.global_iter)
#                        self.tf.add_scalars(main_tag='performance/auc_micro',
#                                            tag_scalar_dict={
#                                                'train_one-shot':auc_micro.item(),
#                                                'train_multi-shot':avg_auc_micro.item()},
#                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/precision_macro',
                                            tag_scalar_dict={
                                                'train_one-shot':precision_macro.item(),
                                                'train_multi-shot':avg_precision_macro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/precision_micro',
                                            tag_scalar_dict={
                                                'train_one-shot':precision_micro.item(),
                                                'train_multi-shot':avg_precision_micro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/precision_fixed_macro',
                                            tag_scalar_dict={
                                                'train_one-shot':precision_fixed_macro.item(),
                                                'train_multi-shot':avg_precision_fixed_macro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/precision_fixed_micro',
                                            tag_scalar_dict={
                                                'train_one-shot':precision_fixed_micro.item(),
                                                'train_multi-shot':avg_precision_fixed_micro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/recall_macro',
                                            tag_scalar_dict={
                                                'train_one-shot':recall_macro.item(),
                                                'train_multi-shot':avg_recall_macro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/recall_micro',
                                            tag_scalar_dict={
                                                'train_one-shot':recall_micro.item(),
                                                'train_multi-shot':avg_recall_micro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/recall_fixed_macro',
                                            tag_scalar_dict={
                                                'train_one-shot':recall_fixed_macro.item(),
                                                'train_multi-shot':avg_recall_fixed_macro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/recall_fixed_micro',
                                            tag_scalar_dict={
                                                'train_one-shot':recall_fixed_micro.item(),
                                                'train_multi-shot':avg_recall_fixed_micro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/f1_macro',
                                            tag_scalar_dict={
                                                'train_one-shot':f1_macro.item(),
                                                'train_multi-shot':avg_f1_macro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/f1_micro',
                                            tag_scalar_dict={
                                                'train_one-shot':f1_micro.item(),
                                                'train_multi-shot':avg_f1_micro.item()
                                                },
                                            global_step=self.global_iter)  
                        self.tf.add_scalars(main_tag='performance/f1_fixed_macro',
                                            tag_scalar_dict={
                                                'train_one-shot':f1_fixed_macro.item(),
                                                'train_multi-shot':avg_f1_fixed_macro.item()
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/f1_fixed_micro',
                                            tag_scalar_dict={
                                                'train_one-shot':f1_fixed_micro.item(),
                                                'train_multi-shot':avg_f1_fixed_micro.item()
                                                },
                                            global_step=self.global_iter)                          
                        self.tf.add_scalars(main_tag='performance/precision_macro_zeropadded',
                                            tag_scalar_dict={
                                                'train_one-shot':precision_macro_zeropadded#,
                                                #'train_multi-shot':avg_precision_macro_zeropadded
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/precision_micro_zeropadded',
                                            tag_scalar_dict={
                                                'train_one-shot':precision_micro_zeropadded#,
                                                #'train_multi-shot':avg_precision_micro_zeropadded
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/recall_macro_zeropadded',
                                            tag_scalar_dict={
                                                'train_one-shot':recall_macro_zeropadded#,
                                                #'train_multi-shot':avg_recall_macro_zeropadded
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/recall_micro_zeropadded',
                                            tag_scalar_dict={
                                                'train_one-shot':recall_micro_zeropadded#,
                                                #'train_multi-shot':avg_recall_micro_zeropadded
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/f1_macro_zeropadded',
                                            tag_scalar_dict={
                                                'train_one-shot':f1_macro_zeropadded#,
                                                #'train_multi-shot':avg_f1_macro_zeropadded
                                                },
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/f1_micro_zeropadded',
                                            tag_scalar_dict={
                                                'train_one-shot':f1_micro_zeropadded#,
                                                #'train_multi-shot':avg_f1_micro_zeropadded
                                                },
                                            global_step=self.global_iter)
                
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={
                                                'train_one-shot_class':class_loss.item(),
                                                'train_one-shot_info':info_loss.item(),
                                                'train_one-shot_total':total_loss.item()},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='mutual_information/train',
                                            tag_scalar_dict={
                                                'I(Z;Y)':izy_bound.item(),
                                                'I(Z;X)':izx_bound.item()},
                                            global_step=self.global_iter)
#%%
            if (self.global_epoch % 2) == 0 : self.scheduler.step()
            self.val()
            
            print("epoch:{}".format(e + 1))
            print('Time spent is {}'.format(time.time() - start)) 
            
        print(" [*] Training Finished!")

    def val(self, test = False):
#%%        
        self.set_mode('eval')
        #self.class_criterion_val = nn.CrossEntropyLoss()#size_average = False)
        #self.info_criterion_val = nn.KLDivLoss()#size_average = False)
        self.class_criterion_val = nn.CrossEntropyLoss(reduction = 'sum')
        self.info_criterion_val = nn.KLDivLoss(reduction = 'sum')
        class_loss = 0
        info_loss = 0
        total_loss = 0
        izy_bound = 0
        izx_bound = 0
        
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
        avg_vmi_fidel_sum = 0
        avg_vmi_fidel_fixed_sum = 0

        correct = 0
        correct_fixed = 0
#        auc_macro = 0
#        auc_micro = 0
#        auc_weighted = 0
        precision_macro = 0  
        precision_micro = 0
        #precision_weighted = 0
        recall_macro = 0
        recall_micro = 0
        #recall_weighted = 0
        f1_macro = 0
        f1_micro = 0
        #f1_weighted = 0
        precision_fixed_macro = 0  
        precision_fixed_micro = 0
        #precision_fixed_weighted = 0
        recall_fixed_macro = 0
        recall_fixed_micro = 0
        #recall_fixed_weighted = 0
        f1_fixed_macro = 0
        f1_fixed_micro = 0
        #f1_fixed_weighted = 0
        avg_correct = 0
        avg_correct_fixed = 0
#        avg_auc_macro = 0
#        avg_auc_micro = 0
#        avg_auc_weighted = 0
        avg_precision_macro = 0  
        avg_precision_micro = 0
        #avg_precision_weighted = 0
        avg_recall_macro = 0
        avg_recall_micro = 0
        #avg_recall_weighted = 0
        avg_f1_macro = 0
        avg_f1_micro = 0
        #avg_f1_weighted = 0
        avg_precision_fixed_macro = 0  
        avg_precision_fixed_micro = 0
        #avg_precision_fixed_weighted = 0
        avg_recall_fixed_macro = 0
        avg_recall_fixed_micro = 0
        #avg_recall_fixed_weighted = 0
        avg_f1_fixed_macro = 0
        avg_f1_fixed_micro = 0
        #avg_f1_fixed_weighted = 0
        total_num = 0
        total_num_ind = 0

#%%     
        with torch.no_grad():
            data_type = 'test' if test else 'valid'  
            #for idx, (x_raw, _, y_raw, _) in enumerate(self.data_loader[data_type]):
            for idx, batch in enumerate(self.data_loader[data_type]):
                  
                if 'mnist' in self.dataset:

                    x_raw = batch[0]
                    y_raw = batch[2]
                    y_ori = batch[1]
                    
                elif 'imdb' in self.dataset:
                    
                    x_raw = batch.text
                    y_raw = batch.label_pred.view(-1)
                    y_ori = batch.label.view(-1) - 2
                
                else:
            
                    raise UnknownDatasetError()
#%%
                
                ## model fit
                x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)
                y_ori = Variable(cuda(y_ori, self.args.cuda)).type(self.y_type)
#%%
                logit, log_p_i, Z_hat, logit_fixed = self.net_ema.model(x)
                #logit, log_p_i, Z_hat, logit_fixed = self.net(x)
                
                ## prior distribution
                p_i_prior = cuda(self.prior(var_size = log_p_i.size()), self.args.cuda)
            
                ## define loss
                y_class = y if len(y.size()) == 1 else torch.argmax(y, dim = -1)
    #            y_binary = label2binary(y_class, classes = range(logit.size(-1)))
    
                class_loss += self.class_criterion_val(logit, y_class).div(math.log(2)) / self.batch_size
                info_loss += self.args.K * self.info_criterion_val(log_p_i, p_i_prior) / self.batch_size
                total_loss += class_loss + self.beta * info_loss
                total_num += 1
                total_num_ind += y_class.size(0)

                prediction = F.softmax(logit, dim=1).max(1)[1]
                correct += torch.eq(prediction, y_class).float().sum()

                prediction_fixed = F.softmax(logit_fixed, dim=1).max(1)[1]
                correct_fixed += torch.eq(prediction_fixed, y_class).float().sum()
                
    #            auc_macro += roc_auc_score(y_binary, logit.detach().numpy(), average = 'macro')
    #            auc_micro += roc_auc_score(y_binary, logit.detach().numpy(), average = 'micro')
    #            auc_weighted += roc_auc_score(y_binary, logit.detach().numpy(), average = 'weighted')    
                precision_macro += precision_score(y_class, prediction, average = 'macro')  
                precision_micro += precision_score(y_class, prediction, average = 'micro')  
                precision_fixed_macro += precision_score(y_class, prediction_fixed, average = 'macro')  
                precision_fixed_micro += precision_score(y_class, prediction_fixed, average = 'micro')
#                precision_weighted += precision_score(y_class, prediction, average = 'weighted')
                recall_macro += recall_score(y_class, prediction, average = 'macro')
                recall_micro += recall_score(y_class, prediction, average = 'micro')
                recall_fixed_macro += recall_score(y_class, prediction_fixed, average = 'macro')
                recall_fixed_micro += recall_score(y_class, prediction_fixed, average = 'micro')
#                recall_weighted += recall_score(y_class, prediction, average = 'weighted')
                f1_macro += f1_score(y_class, prediction, average = 'macro')
                f1_micro += f1_score(y_class, prediction, average = 'micro')
                f1_fixed_macro += f1_score(y_class, prediction_fixed, average = 'macro')
                f1_fixed_micro += f1_score(y_class, prediction_fixed, average = 'micro')
#                f1_weighted += f1_score(y_class, prediction, average = 'weighted')

                # selected chunk index
                _, index_chunk = log_p_i.unsqueeze(1).topk(self.args.K, dim = -1)

                if self.chunk_size is not 1:
                    
                    index_chunk = index_transfer(dataset = self.dataset,
                                                 idx = index_chunk, 
                                                 filter_size = self.filter_size,
                                                 original_nrow = self.original_nrow,
                                                 original_ncol = self.original_ncol, 
                                                 is_cuda = self.cuda).output
#%%               
                if 'mnist' in self.dataset:
                
                    data_size = x_raw.size()
                    binary_selected_all = idxtobool(index_chunk.view(data_size[0], data_size[1], -1), [data_size[0], data_size[1], data_size[2] * data_size[3]], self.cuda)            
                    data_zeropadded = torch.addcmul(torch.zeros(1), value=1, tensor1=binary_selected_all.view(data_size).type(torch.FloatTensor), tensor2=x_raw.type(torch.FloatTensor), out=None)
                    data_zeropadded = data_zeropadded.type(self.x_type)
                    #data_zeropadded[data_zeropadded == 0] = -1
                    
                elif 'imdb' in self.dataset:
                
                    data_size = x_raw.size()
                    binary_selected_all = idxtobool(index_chunk.view(data_size[0], -1), [data_size[0], data_size[1]], self.cuda)            
                    data_zeropadded = torch.addcmul(torch.zeros(1), value=1, tensor1=binary_selected_all.view(data_size).type(torch.FloatTensor), tensor2=x_raw.type(torch.FloatTensor), out=None)
                    data_zeropadded = data_zeropadded.type(self.x_type)
                    data_zeropadded[data_zeropadded == 0] = 1
                
                else:
                
                    raise UnknownDatasetError()
#%%        
                # Post-hoc Accuracy (zero-padded accuracy)
                #output_zeropadded, _, _, _ = self.net_ema.model(data_zeropadded)
                output_original = self.black_box(x)
                output_zeropadded = self.black_box(data_zeropadded)    
                pred_zeropadded = F.softmax(output_zeropadded, dim=-1).max(1)[1]
                #pred_zeropadded = output_zeropadded.max(1, keepdim=True)[1] 
                correct_zeropadded += pred_zeropadded.eq(y_class).float().sum()                
                precision_macro_zeropadded += precision_score(y_class, pred_zeropadded, average = 'macro')  
                precision_micro_zeropadded += precision_score(y_class, pred_zeropadded, average = 'micro')  
                precision_weighted_zeropadded += precision_score(y_class, pred_zeropadded, average = 'weighted')
                recall_macro_zeropadded += recall_score(y_class, pred_zeropadded, average = 'macro')
                recall_micro_zeropadded += recall_score(y_class, pred_zeropadded, average = 'micro')
                recall_weighted_zeropadded += recall_score(y_class, pred_zeropadded, average = 'weighted')
                f1_macro_zeropadded += f1_score(y_class, pred_zeropadded, average = 'macro')
                f1_micro_zeropadded += f1_score(y_class, pred_zeropadded, average = 'micro')
                f1_weighted_zeropadded += f1_score(y_class, pred_zeropadded, average = 'weighted')
                
                if idx == 0:
                        
                    print("seojin")
                    print("zeropadded", pred_zeropadded)
                    print("bb-pred", y_class)
                    print()
        
                ## Variational Mutual Information                           
                vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                              tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                              tensor2 = output_zeropadded.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(output_zeropadded.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                              out=None), dim = -1)
                vmi_zeropadded_sum += vmi.sum()
                vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                              tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                              tensor2 = logit.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(logit.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                              out=None), dim = -1)
                vmi_fidel_sum += vmi.sum()

                vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                              tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                              tensor2 = logit_fixed.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(logit_fixed.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                              out=None), dim = -1)
                vmi_fidel_fixed_sum += vmi.sum()

                
#%%                
                if self.num_avg != 0 :
                    #print("multishot")
                    avg_soft_logit, avg_log_p_i, _, avg_soft_logit_fixed = self.net_ema.model(x, self.num_avg)
                    #avg_soft_logit, _, _, avg_soft_logit_fixed = self.net(x,self.num_avg)
                    avg_prediction = avg_soft_logit.max(1)[1]
                    avg_correct += torch.eq(avg_prediction,y_class).float().sum()
                    avg_prediction_fixed  = avg_soft_logit_fixed.max(1)[1]
                    avg_correct_fixed  += torch.eq(avg_prediction_fixed,y_class).float().sum()
    #                #avg_prediction = avg_soft_logit.max(1)[1]
    #                avg_prediction = torch.argmax(avg_soft_logit, dim = -1)
    #                avg_correct += torch.eq(avg_prediction,y).float().sum()
    #                avg_auc_macro += roc_auc_score(y_binary, avg_soft_logit.detach().numpy(), average = 'macro')
    #                avg_auc_micro += roc_auc_score(y_binary, avg_soft_logit.detach().numpy(), average = 'micro')
    #                avg_auc_weighted += roc_auc_score(y_binary, avg_soft_logit.detach().numpy(), average = 'weighted') 
                    avg_precision_macro += precision_score(y_class, avg_prediction, average = 'macro')  
                    avg_precision_micro += precision_score(y_class, avg_prediction, average = 'micro')  
                    #avg_precision_weighted += precision_score(y_class, avg_prediction, average = 'weighted')
                    avg_recall_macro += recall_score(y_class, avg_prediction, average = 'macro')
                    avg_recall_micro += recall_score(y_class, avg_prediction, average = 'micro')
                    #avg_recall_weighted += recall_score(y_class, avg_prediction, average = 'weighted')
                    avg_f1_macro += f1_score(y_class, avg_prediction, average = 'macro')
                    avg_f1_micro += f1_score(y_class, avg_prediction, average = 'micro')
                    #avg_f1_weighted += f1_score(y_class, avg_prediction, average = 'weighted') 
                    
                    avg_precision_fixed_macro += precision_score(y_class, avg_prediction_fixed, average = 'macro')  
                    avg_precision_fixed_micro += precision_score(y_class, avg_prediction_fixed, average = 'micro')  
                    #avg_precision_fixed_weighted += precision_score(y_class, avg_prediction_fixed, average = 'weighted')
                    avg_recall_fixed_macro += recall_score(y_class, avg_prediction_fixed, average = 'macro')
                    avg_recall_fixed_micro += recall_score(y_class, avg_prediction_fixed, average = 'micro')
                    #avg_recall_fixed_weighted += recall_score(y_class, avg_prediction_fixed, average = 'weighted')
                    avg_f1_fixed_macro += f1_score(y_class, avg_prediction_fixed, average = 'macro')
                    avg_f1_fixed_micro += f1_score(y_class, avg_prediction_fixed, average = 'micro')
                    #avg_f1_fixed_weighted += f1_score(y_class, avg_prediction_fixed, average = 'weighted') 
                    
                    ## Variational Mutual Information            
                    vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                                  tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                                  tensor2 = avg_soft_logit.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(avg_soft_logit.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                                  out=None), dim = -1)
                    avg_vmi_fidel_sum += vmi.sum()
    
                    vmi = torch.sum(torch.addcmul(torch.zeros(1), value = 1, 
                                                  tensor1 = torch.exp(output_original).type(torch.FloatTensor),
                                                  tensor2 = avg_soft_logit_fixed.type(torch.FloatTensor) - torch.logsumexp(output_original, dim = 0).unsqueeze(0).expand(avg_soft_logit_fixed.size()).type(torch.FloatTensor) + torch.log(torch.tensor(output_original.size(0)).type(torch.FloatTensor)),
                                                  out=None), dim = -1)
                    avg_vmi_fidel_fixed_sum += vmi.sum()
                    
                else :
    #                avg_correct = Variable(cuda(torch.zeros(correct.size()), self.args.cuda))
    #                avg_correct_fixed = Variable(cuda(torch.zeros(correct_fixed.size()), self.args.cuda))
    ##                avg_auc_macro = Variable(cuda(torch.zeros(auc_macro.size()), self.args.cuda))
    ##                avg_auc_micro = Variable(cuda(torch.zeros(auc_micro.size()), self.args.cuda))
    ##                avg_auc_weighted = Variable(cuda(torch.zeros(auc_weighted.size()), self.args.cuda))
    #                avg_precision_macro = Variable(cuda(torch.zeros(precision_macro.size()), self.args.cuda))
    #                avg_precision_micro = Variable(cuda(torch.zeros(precision_micro.size()), self.args.cuda))
    #                avg_precision_weighted = Variable(cuda(torch.zeros(precision_weighted.size()), self.args.cuda))
    #                avg_recall_macro = Variable(cuda(torch.zeros(recall_macro.size()), self.args.cuda))
    #                avg_recall_micro = Variable(cuda(torch.zeros(recall_micro.size()), self.args.cuda))
    #                avg_recall_weighted = Variable(cuda(torch.zeros(recall_weighted.size()), self.args.cuda))
    #                avg_f1_macro = Variable(cuda(torch.zeros(f1_macro.size()), self.args.cuda))
    #                avg_f1_micro = Variable(cuda(torch.zeros(f1_micro.size()), self.args.cuda))
    #                avg_f1_weighted = Variable(cuda(torch.zeros(f1_weighted.size()), self.args.cuda))
                    avg_correct = correct
                    avg_correct_fixed = correct_fixed
                    #avg_auc_macro = auc_macro
                    #avg_auc_micro = auc_micro
                    #avg_auc_weighted = auc_weighted
                    avg_precision_macro = precision_macro
                    avg_precision_micro = precision_micro
                    #avg_precision_weighted = precision_weighted
                    avg_recall_macro = recall_macro
                    avg_recall_micro = recall_micro
                    #avg_recall_weighted = recall_weighted
                    avg_f1_macro = f1_macro
                    avg_f1_micro = f1_micro
                    #avg_f1_weighted = f1_weighted
                    
                    avg_recall_fixed_macro = recall_fixed_macro
                    avg_recall_fixed_micro = recall_fixed_micro
                    #avg_recall_fixed_weighted = recall_fixed_weighted
                    avg_f1_fixed_macro = f1_fixed_macro
                    avg_f1_fixed_micro = f1_fixed_micro
                    #avg_f1_fixed_weighted = f1_fixed_weighted
                    
                    avg_vmi_fidel_sum = vmi_fidel_sum
                    avg_vmi_fidel_fixed_sum = vmi_fidel_fixed_sum
                
                #%% save image #
                if (self.global_iter > 4999 and self.global_epoch % 5 == 0) or self.global_epoch is self.epoch:
                    #print("SAVED!!!!")
                    if (not test) and (idx == 0 or idx == 200):
        
                        # filename
                        img_name, _ = os.path.splitext(self.checkpoint_name)
                        img_name = 'figure_' + img_name + '_' + str(self.global_epoch) + "_" + str(idx) + '.png'
                        img_name = Path(self.image_dir).joinpath(img_name)
                        '''
                        # selected chunk index
                        _, index_chunk = log_p_i.unsqueeze(1).topk(self.args.K, dim = -1)
        
                        if self.chunk_size is not 1:
                            
                            index_chunk = index_transfer(dataset = self.dataset,
                                                         idx = index_chunk, 
                                                         filter_size = self.filter_size,
                                                         original_nrow = self.original_nrow,
                                                         original_ncol = self.original_ncol, 
                                                         is_cuda = self.cuda).output
                        '''
                        save_batch(dataset = self.dataset, 
                                   batch = x, 
                                   label = y_ori, label_pred = y_class, label_approx = prediction,
                                   index = index_chunk, 
                                   filename = img_name, 
                                   is_cuda = self.cuda,
                                   word_idx = self.args.word_idx).output##
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

            vmi_zeropadded = vmi_zeropadded_sum/total_num_ind
            vmi_fidel = vmi_fidel_sum/total_num_ind
            vmi_fidel_fixed = vmi_fidel_fixed_sum/total_num_ind
            avg_vmi_fidel = avg_vmi_fidel_sum/total_num_ind
            avg_vmi_fidel_fixed = avg_vmi_fidel_fixed_sum/total_num_ind
            
            ## Approximation Fidelity (prediction performance)            
            accuracy = correct/total_num_ind
            avg_accuracy = avg_correct/total_num_ind
            accuracy_fixed  = correct_fixed/total_num_ind
            avg_accuracy_fixed  = avg_correct_fixed  / total_num_ind
    #        auc_macro = auc_macro/total_num
    #        auc_micro = auc_micro/total_num
    #        auc_weighted = auc_weighted/total_num
            precision_macro = precision_macro/total_num
            precision_micro = precision_micro/total_num
            #precision_weighted = precision_weighted/total_num
            recall_macro = recall_macro/total_num
            recall_micro = recall_micro/total_num
            #recall_weighted = recall_weighted/total_num
            f1_macro = f1_macro/total_num
            f1_micro = f1_micro/total_num
            #f1_weighted = f1_weighted/total_num
            
            precision_fixed_macro = precision_fixed_macro/total_num
            precision_fixed_micro = precision_fixed_micro/total_num
            #precision_fixed_weighted = precision_fixed_weighted/total_num
            recall_fixed_macro = recall_fixed_macro/total_num
            recall_fixed_micro = recall_fixed_micro/total_num
            #recall_fixed_weighted = recall_fixed_weighted/total_num
            f1_fixed_macro = f1_fixed_macro/total_num
            f1_fixed_micro = f1_fixed_micro/total_num
            #f1_fixed_weighted = f1_fixed_weighted/total_num
            
    #        avg_auc_macro = avg_auc_macro/total_num
    #        avg_auc_micro = avg_auc_micro/total_num
    #        avg_auc_weighted = avg_auc_weighted/total_num
            avg_precision_macro = avg_precision_macro/total_num
            avg_precision_micro = avg_precision_micro/total_num
            #avg_precision_weighted = avg_precision_weighted/total_num
            avg_recall_macro = avg_recall_macro/total_num
            avg_recall_micro = avg_recall_micro/total_num
            #avg_recall_weighted = avg_recall_weighted/total_num
            avg_f1_macro = avg_f1_macro/total_num
            avg_f1_micro = avg_f1_micro/total_num
            #avg_f1_weighted = avg_f1_weighted/total_num
            
            avg_precision_fixed_macro = avg_precision_fixed_macro/total_num
            avg_precision_fixed_micro = avg_precision_fixed_micro/total_num
            #avg_precision_fixed_weighted = avg_precision_fixed_weighted/total_num
            avg_recall_fixed_macro = avg_recall_fixed_macro/total_num
            avg_recall_fixed_micro = avg_recall_fixed_micro/total_num
            #avg_recall_fixed_weighted = avg_recall_fixed_weighted/total_num
            avg_f1_fixed_macro = avg_f1_fixed_macro/total_num
            avg_f1_fixed_micro = avg_f1_fixed_micro/total_num
            #avg_f1_fixed_weighted = avg_f1_fixed_weighted/total_num

            class_loss /= total_num
            info_loss /= total_num
            total_loss /= total_num
            izy_bound = math.log(10,2) - class_loss
            izx_bound = info_loss
            
            print('\n\n[VAL RESULT]\n')
            #tab = pd.crosstab(y_class, prediction)
            #print(tab, end = "\n")
            print('epoch {}'.format(self.global_epoch), end = "\n") 
            print('global iter {}'.format(self.global_iter), end = "\n")                   
            print('IZY:{:.2f} IZX:{:.2f}'
                    .format(izy_bound.item(), izx_bound.item()), end = '\n')
            print('acc:{:.4f} avg_acc:{:.4f}'
                    .format(accuracy.item(), avg_accuracy.item()), end = '\n')          
            print('acc_fixed:{:.4f} avg_acc_fixed:{:.4f}'
                    .format(accuracy_fixed.item(), avg_accuracy_fixed.item()), end='\n')
            print('vmi:{:.4f} avg_vmi:{:.4f}'
                    .format(vmi_fidel.item(), avg_vmi_fidel.item()), end = '\n')
            print('vmi_fixed:{:.4f} avg_vmi_fixed:{:.4f}'
                    .format(vmi_fidel_fixed.item(), avg_vmi_fidel_fixed.item()), end = '\n')
            print('acc_zeropadded:{:.4f} vmi_zeropadded:{:.4f}'
                    .format(accuracy_zeropadded.item(), vmi_zeropadded.item()), end = '\n')                 
#    #        print('auc_macro:{:.4f} avg_auc_macro:{:.4f}'
#    #                .format(auc_macro.item(), avg_auc_macro.item()), end = '\n')   
#    #        print('auc_micro:{:.4f} avg_auc_micro:{:.4f}'
#    #                .format(auc_micro.item(), avg_auc_micro.item()), end = '\n')     
#            print('precision_macro:{:.4f} avg_precision_macro:{:.4f}'
#                    .format(precision_macro, avg_precision_macro), end = '\n') 
#            print('precision_micro:{:.4f} avg_precision_micro:{:.4f}'
#                    .format(precision_micro, avg_precision_micro), end = '\n')
#            print('recall_macro:{:.4f} avg_recall_macro:{:.4f}'
#                    .format(recall_macro, avg_recall_macro), end = '\n')   
#            print('recall_micro:{:.4f} avg_recall_micro:{:.4f}'
#                    .format(recall_micro, avg_recall_micro), end = '\n') 
#            print('f1_macro:{:.4f} avg_f1_macro:{:.4f}'
#                    .format(f1_macro, avg_f1_macro), end = '\n')   
#            print('f1_micro:{:.4f} avg_f1_micro:{:.4f}'
#                    .format(f1_micro, avg_f1_micro), end = '\n') 
#            print('precision_macro_zeropadded:{:.4f}'
#                    .format(precision_macro_zeropadded), end = '\n')
#            print('precision_micro_zeropadded:{:.4f}'
#                    .format(precision_micro_zeropadded), end = '\n')
#            print('recall_macro_zeropadded:{:.4f}'
#                    .format(recall_macro_zeropadded), end = '\n')   
#            print('recall_micro_zeropadded:{:.4f}'
#                    .format(recall_micro_zeropadded), end = '\n') 
#            print('f1_macro_zeropadded:{:.4f}'
#                    .format(f1_macro_zeropadded), end = '\n')   
#            print('f1_micro_zeropadded:{:.4f}'
#                    .format(f1_micro_zeropadded), end = '\n') 
            
            print()
#%%                
            if self.history['avg_acc'] < avg_accuracy.item() :
                
                self.history['class_loss'] = class_loss.item()
                self.history['info_loss'] = info_loss.item()
                self.history['total_loss'] = total_loss.item()
                self.history['epoch'] = self.global_epoch
                self.history['iter'] = self.global_iter

                self.history['avg_acc'] = avg_accuracy.item()
                self.history['avg_acc_fixed'] = avg_accuracy_fixed.item()
    #            self.history['avg_auc_macro'] = avg_auc_macro.item()
    #            self.history['avg_auc_micro'] = avg_auc_micro.item()
    #            self.history['avg_auc_weighted'] = avg_auc_weighted.item()
                self.history['avg_precision_macro'] = avg_precision_macro
                self.history['avg_precision_micro'] = avg_precision_micro
                #self.history['avg_precision_weighted'] = avg_precision_weighted
                self.history['avg_recall_macro'] = avg_recall_macro
                self.history['avg_recall_micro'] = avg_recall_micro
                #self.history['avg_recall_weighted'] = avg_recall_weighted
                self.history['avg_f1_macro'] = avg_f1_macro
                self.history['avg_f1_micro'] = avg_f1_micro
                #self.history['avg_f1_weighted'] = avg_f1_weighted

                self.history['avg_precision_fixed_macro'] = avg_precision_fixed_macro
                self.history['avg_precision_fixed_micro'] = avg_precision_fixed_micro
                #self.history['avg_precision_fixed_weighted'] = avg_precision_fixed_weighted
                self.history['avg_recall_fixed_macro'] = avg_recall_fixed_macro
                self.history['avg_recall_fixed_micro'] = avg_recall_fixed_micro
                #self.history['avg_recall_fixed_weighted'] = avg_recall_fixed_weighted
                self.history['avg_f1_fixed_macro'] = avg_f1_fixed_macro
                self.history['avg_f1_fixed_micro'] = avg_f1_fixed_micro
                #self.history['avg_f1_fixed_weighted'] = avg_f1_fixed_weighted
                
                self.history['acc_zeropadded'] = accuracy_zeropadded.item()
                self.history['precision_macro_zeropadded'] = precision_macro_zeropadded.item()
                self.history['precision_micro_zeropadded'] = precision_micro_zeropadded.item()
                #self.history['precision_weighted_zeropadded'] = precision_weighted_zeropadded.item()
                self.history['recall_macro_zeropadded'] = recall_macro_zeropadded.item()
                self.history['recall_micro_zeropadded'] = recall_micro_zeropadded.item()
#                self.history['recall_weighted_zeropadded'] = recall_weighted_zeropadded.item()
                self.history['f1_macro_zeropadded'] = f1_macro_zeropadded.item()
                self.history['f1_micro_zeropadded'] = f1_micro_zeropadded.item()
#                self.history['f1_weighted_zeropadded'] = f1_weighted_zeropadded.item()
                self.history['vmi_zeropadded'] = vmi_zeropadded.item()
                self.history['avg_vmi'] = avg_vmi_fidel.item()
                self.history['avg_vmi_fixed'] = avg_vmi_fidel_fixed.item()            
            
                # if save_checkpoint : self.save_checkpoint('best_acc.tar')
                if not test : self.save_checkpoint(self.checkpoint_name)
                
            if self.tensorboard :
                self.tf.add_scalars(main_tag='performance/accuracy',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':accuracy.item(),
                                        data_type + '_multi-shot':avg_accuracy.item()
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/accuracy_fixed',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':accuracy_fixed.item(),
                                        data_type + '_multi-shot':avg_accuracy_fixed.item()
                                        },
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/vmi',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':vmi_fidel.item(),
                                        data_type + '_multi-shot':avg_vmi_fidel.item()
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/vmi_fixed',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':vmi_fidel_fixed.item(),
                                        data_type + '_multi-shot':avg_vmi_fidel_fixed.item()
                                        },
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/accuracy_zeropadded',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':accuracy_zeropadded.item()#,
                                        #data_type + '_multi-shot':accuracy_zeropadded.item()
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/vmi_zeropadded',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':vmi_zeropadded.item()#,
                                        #'train_multi-shot':avg_accuracy_zeropadded.item()
                                        },
                                    global_step=self.global_iter)    
#                self.tf.add_scalars(main_tag='performance/error',
#                                    tag_scalar_dict={
#                                        data_type + '_one-shot':1-accuracy.item(),
#                                        data_type + '_multi-shot':1-avg_accuracy.item()
#                                        },
#                                    global_step=self.global_iter)
    #            self.tf.add_scalars(main_tag='performance/auc_macro',
    #                                tag_scalar_dict={
    #                                    data_type + '_one-shot':auc_macro.item(),
    #                                    data_type + '_multi-shot':avg_auc_macro.item()},
    #                                global_step=self.global_iter)
    #            self.tf.add_scalars(main_tag='performance/auc_micro',
    #                                tag_scalar_dict={
    #                                    data_type + '_one-shot':auc_micro.item(),
    #                                    data_type + '_multi-shot':avg_auc_micro.item()},
    #                                global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/precision_macro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':precision_macro,
                                        data_type + '_multi-shot':avg_precision_macro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/precision_micro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':precision_micro,
                                        data_type + '_multi-shot':avg_precision_micro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/recall_macro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':recall_macro,
                                        data_type + '_multi-shot':avg_recall_macro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/recall_micro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':recall_micro,
                                        data_type + '_multi-shot':avg_recall_micro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/f1_macro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':f1_macro,
                                        data_type + '_multi-shot':avg_f1_macro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/f1_micro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':f1_micro,
                                        data_type + '_multi-shot':avg_f1_micro
                                        },
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/precision_fixed_macro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':precision_fixed_macro,
                                        data_type + '_multi-shot':avg_precision_fixed_macro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/precision_fixed_micro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':precision_fixed_micro,
                                        data_type + '_multi-shot':avg_precision_fixed_micro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/recall_fixed_macro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':recall_fixed_macro,
                                        data_type + '_multi-shot':avg_recall_fixed_macro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/recall_fixed_micro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':recall_fixed_micro,
                                        data_type + '_multi-shot':avg_recall_fixed_micro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/f1_fixed_macro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':f1_fixed_macro,
                                        data_type + '_multi-shot':avg_f1_fixed_macro
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/f1_fixed_micro',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':f1_fixed_micro,
                                        data_type + '_multi-shot':avg_f1_fixed_micro
                                        },
                                    global_step=self.global_iter)
                
                self.tf.add_scalars(main_tag='performance/precision_macro_zeropadded',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':precision_macro_zeropadded#,
                                        #data_type + '_multi-shot':precision_macro_zeropadded
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/precision_micro_zeropadded',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':precision_micro_zeropadded#,
                                        #data_type + '_multi-shot':precision_micro_zeropadded
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/recall_macro_zeropadded',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':recall_macro_zeropadded#,
                                        #data_type + '_multi-shot':recall_macro_zeropadded
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/recall_micro_zeropadded',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':recall_micro_zeropadded#,
                                        #data_type + '_multi-shot':recall_micro_zeropadded
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/f1_macro_zeropadded',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':f1_macro_zeropadded#,
                                        #data_type + '_multi-shot':f1_macro_zeropadded
                                        },
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='performance/f1_micro_zeropadded',
                                    tag_scalar_dict={
                                        data_type + '_one-shot':f1_micro_zeropadded#,
                                        #data_type + '_multi-shot':f1_micro_zeropadded
                                        },
                                    global_step=self.global_iter)
                
                self.tf.add_scalars(main_tag='performance/cost',
                                    tag_scalar_dict={
                                        data_type + '_one-shot_class':class_loss.item(),
                                        data_type + '_one-shot_info':info_loss.item(),
                                        data_type + '_one-shot_total':total_loss.item()},
                                    global_step=self.global_iter)
                self.tf.add_scalars(main_tag='mutual_information/val',
                                    tag_scalar_dict={
                                        'I(Z;Y)':izy_bound.item(),
                                        'I(Z;X)':izx_bound.item()},
                                    global_step=self.global_iter)

        self.set_mode('train')

    def save_checkpoint(self, filename='best_acc.tar'):
        model_states = {
                'net':self.net.state_dict(),
                'net_ema':self.net_ema.model.state_dict(),
                }
        optim_states = {
                'optim':self.optim.state_dict(),
                }
        states = {
                'iter':self.global_iter,
                'epoch':self.global_epoch,
                'history':self.history,
                'args':self.args,
                'model_states':model_states,
                'optim_states':optim_states,
                }

        file_path = self.checkpoint_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))

        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        
        file_path = self.checkpoint_dir.joinpath(filename)
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.net_ema.model.load_state_dict(checkpoint['model_states']['net_ema'])

            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))

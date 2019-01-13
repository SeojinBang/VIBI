#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from utils import Flatten, idxtobool, TimeDistributed, Concatenate, UnknownModelError
#%%
class Explainer(nn.Module):

    def __init__(self, **kwargs):
        
        super(Explainer, self).__init__()

        self.args = kwargs['args']
        self.mode = self.args.mode
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.tau = 0.5 # float, parameter for concrete random variable distribution
        self.K = self.args.K # number of variables selected
        self.approximater_type = self.args.approximater_type if self.args.approximater_type is not 'None' else 'lstm'
        self.chunk_size = self.args.chunk_size

        self.max_total_num_words = self.args.max_total_num_words
        self.embedding_dim = self.args.embedding_dim
        self.max_num_words = self.args.max_num_words #100
        self.max_num_sents = self.args.max_num_sents #15
        
        ## Basic Model Structure
        
        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings = self.max_total_num_words + 2, 
                                            embedding_dim = self.embedding_dim) 

        ## explainer 
        if self.chunk_size == self.max_num_words:

            # parameters for explainer
            self.hidden_size = 250
            self.hidden2_size = 100
            self.hidden_global_size = 100
            self.hidden_local_size = 50

            # model structure for explainer
            self.sent_encoder = nn.Sequential(
                    nn.Dropout(p = 0.2), 
                    nn.Conv1d(in_channels = self.embedding_dim, out_channels = self.hidden_size, kernel_size = 3, padding = 0, stride = 1), # torch.Size([150, 250, 98])
                    nn.ReLU(True), # torch.Size([15, 250, 98])
                    nn.MaxPool1d(kernel_size = self.max_num_words - 3 + 1) # torch.Size([150, 250, 1])
                    )
            self.sent_encoder2 = nn.Sequential(
                    nn.Conv1d(in_channels = self.hidden_size, out_channels = self.hidden2_size, kernel_size = 3, padding = 1, stride = 1),
                    nn.ReLU(True),
                    ) # torch.Size([150, 100, 1])
            self.sent_encoder_global = nn.Sequential( 
                                        nn.Linear(in_features = self.hidden2_size, out_features = self.hidden_global_size),
                                        nn.ReLU(True)
                                        ) # torch.Size([10, 100])  
            self.sent_encoder_local = nn.Sequential(
                    nn.Conv1d(in_channels = self.hidden_size, out_channels = 50, kernel_size = 3, padding = 1, stride = 1),
                    nn.ReLU(True),
                    nn.Conv1d(in_channels = 50, out_channels = self.hidden_local_size, kernel_size = 3, padding = 1, stride = 1),
                    nn.ReLU(True)
                    )  
            self.sent_encoder_combined = nn.Sequential(
                        nn.Dropout(p = 0.2),
                        nn.Conv1d(in_channels = self.hidden_global_size + self.hidden_local_size, out_channels = 50, kernel_size = 1, padding = 0, stride = 1),
                        nn.ReLU(True),
                        nn.Conv1d(in_channels = 50, out_channels = 1, kernel_size = 1, padding = 0, stride = 1),
                        nn.LogSoftmax(-1)
                        )
            self.sent_decoder = nn.Sequential(
                        nn.Linear(in_features = self.hidden_size, out_features = 250),
                        # nn.Dropout(p = 0.2),
                        nn.ReLU(True), # [10, 250]
                        nn.Linear(in_features = 250, out_features = 2), # [10, 2]
                        nn.LogSoftmax(1)
                        )
            
        if self.chunk_size == 1:
            
            # parameters for explainer
            self.bidirectional = True
            self.num_wordlevel_explainer = 50
            self.num_wordlevel_explainer = self.num_wordlevel_explainer // 2 if self.bidirectional else self.num_wordlevel_explainer
            
            self.word_level = nn.LSTM(input_size = self.args.embedding_dim, 
                                      hidden_size = self.num_wordlevel_explainer, 
                                      num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
            self.concat = nn.LogSoftmax(-1)

        else:
            # parameters for explainer
            self.num_wordlevel_explainer = 50 // 2
            
            # model structure for explainer 
            self.word_level = nn.Sequential(
                    nn.Dropout(p = 0.2),
                    nn.Conv1d(in_channels = 1, out_channels = self.num_wordlevel_explainer,
                            kernel_size = self.embedding_dim * self.chunk_size, #self.chunk_size,
                            padding = 0, #self.embedding_dim * (padding_size // 2),
                            stride = self.embedding_dim)
                    )       
            self.sent_level = nn.Sequential(
                    nn.ReLU(True),
                    nn.MaxPool1d(kernel_size = self.max_num_words - self.chunk_size + 1, stride = 1))
            self.concat_level = nn.Sequential(
                    #nn.Dropout(p = 0.2),
                    nn.Linear(in_features = self.num_wordlevel_explainer * 2, out_features = 1),
                    #nn.ReLU(True),
                    #nn.Linear(in_features = 50, out_features = 1)
                    nn.LogSoftmax(1)
                    )

        ## paramters for approximater
        self.num_rnn_out_size = 50
        self.num_cnn_hidden_size = 100
        self.num_cnn_kernel_size = 3
        self.bidirectional = True  #
        self.mul_bidirect = 2 if self.bidirectional else 1

        self.dense = nn.Linear(self.max_num_sents * 2 * self.mul_bidirect, 2)
        
        ## model structure for approximater
        if self.approximater_type in ['rnn', 'RNN']:

            self.RNN_word = nn.RNN(input_size = self.args.embedding_dim, hidden_size = self.num_rnn_out_size,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
    
            self.RNN_sent = nn.RNN(input_size = self.num_rnn_out_size * self.mul_bidirect, hidden_size = 2,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
        
            self.approximater_type = 'rnn'
            self.encoder_word = self.RNN_word
            self.encoder_sent = self.RNN_sent

        elif self.approximater_type in ['lstm', 'LSTM']:
    
            self.LSTM_word = nn.LSTM(input_size = self.args.embedding_dim, hidden_size = self.num_rnn_out_size,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
    
            self.LSTM_sent = nn.LSTM(input_size = self.num_rnn_out_size * self.mul_bidirect, hidden_size = 2,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
        
            self.approximater_type = 'lstm'
            self.encoder_word = self.LSTM_word
            self.encoder_sent = self.LSTM_sent

        elif self.approximater_type in ['lstm-light', 'LSTM-light', 'LSTM-Light', 'LSTM-LIGHT']:
            
            self.LSTM_word = nn.LSTM(input_size = self.args.embedding_dim, hidden_size = 1,
                    num_layers = 1, batch_first = True, bidirectional = True)

            self.approximater_type = 'lstm-light'
            self.encoder_word = self.LSTM_word
            self.encoder_sent = nn.Linear(self.max_num_sents * self.max_num_words * self.mul_bidirect, 2)

        elif self.approximater_type in ['lstm-light-onedirect', 'LSTM-light-onedirect', 'LSTM-Light-onedirect', 'LSTM-LIGHT-onedirect']:
            
            self.LSTM_word = nn.LSTM(input_size = self.args.embedding_dim, hidden_size = 1,
                    num_layers = 1, batch_first = True, bidirectional = False)

            self.approximater_type = 'lstm-light'
            self.encoder_word = self.LSTM_word
            self.encoder_sent = nn.Linear(self.max_num_sents * self.max_num_words, 2)
            
        elif self.approximater_type in ['gru', 'GRU']:
            
            self.GRU_word = nn.GRU(input_size = self.args.embedding_dim, hidden_size = self.num_rnn_out_size,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
    
            self.GRU_sent = nn.GRU(input_size = self.num_rnn_out_size * self.mul_bidirect, hidden_size = 2,
                    num_layers = 1, batch_first = True, bidirectional = self.bidirectional)
        
            self.approximater_type = 'gru'
            self.encoder_word = self.GRU_word
            self.encoder_sent = self.GRU_sent

        elif self.approximater_type in ['cnn', 'CNN']:
            
            self.CNN_word = nn.Sequential(
                    nn.Dropout(p = 0.2),
                    nn.Conv1d(in_channels = 1, out_channels = self.num_cnn_hidden_size, kernel_size = self.embedding_dim * self.num_cnn_kernel_size, padding = 0, stride = self.embedding_dim), # torch.Size([150, 250, 98])
                    nn.ReLU(True),
                    nn.MaxPool1d(kernel_size = self.max_num_words - self.num_cnn_kernel_size + 1)
                    )
    
            self.CNN_sent = nn.Sequential(
                        nn.Linear(in_features = self.num_cnn_hidden_size, out_features = 2),
                        nn.LogSoftmax(1)
                        )
        
            self.approximater_type = 'cnn'
            self.encoder_word = self.CNN_word
            self.encoder_sent = self.CNN_sent

        else:
            
            raise UnknownModelError()
        
          
    def explainer(self, x):
        
        embeded_words = self.embedding_layer(x).view(-1, self.max_num_words, self.embedding_dim) ## embedding [batch * sent, word, dim]
        
        if self.chunk_size == self.max_num_words: 

            ## sentence encoder
            encoded_sent = TimeDistributed(self.sent_encoder)(embeded_words) # torch.Size([1500, 1, 250]) [batch-size * sent-num, 1, self.hidden_size]
            encoded_sent = encoded_sent.view(-1, self.max_num_sents, 250)# torch.Size([100, 15, 250]) [batch-size, sent-num, self.hidden_size]
        
            ## first layer
            first_layer= TimeDistributed(self.sent_encoder2)(encoded_sent) # torch.Size([100, 15, 100])
         
            ## global info
            encoded_review_global = nn.MaxPool1d(kernel_size = self.max_num_sents)(first_layer.permute(0, 2, 1)).squeeze(2) # torch.Size([10, 100, 1])
            encoded_review_global = self.sent_encoder_global(encoded_review_global)

            ## local info
            encoded_review_local = TimeDistributed(self.sent_encoder_local)(encoded_sent)

            ## combined 
            combined = Concatenate(encoded_review_global, encoded_review_local) # should be (10, 15, 150)
            logits_T = TimeDistributed(self.sent_encoder_combined)(combined).squeeze(-1) # orch.Size([100, 15])

        elif self.chunk_size == 1:

            encoded_words = embeded_words.view(-1, self.max_num_sents * self.max_num_words, self.embedding_dim, 1).squeeze(-1)
            encoded_words = self.word_level(encoded_words)[0] 
            encoded_words = torch.mean(encoded_words, -1)
            
            logits_T = self.concat(encoded_words)#.view(-1, self.max_num_sents, self.max_num_words)
          
        else:
            
            encoded_words = self.word_level(embeded_words.view(-1, 1, self.max_num_words * self.embedding_dim))
            encoded_sent = self.sent_level(encoded_words).view(-1, 1, self.num_wordlevel_explainer).expand(encoded_words.size(0), encoded_words.size(2), encoded_words.size(1)) ## 750, 46, 25
            logits_T = self.concat_level(torch.cat((encoded_words.permute(0, 2, 1), encoded_sent), -1)).squeeze(-1).view(-1, self.max_num_sents * encoded_sent.size(-2))

#            ### word level 
#            encoded_words0 = self.word_level(embeded_words.view(-1, 1, self.max_num_words * self.embedding_dim))
#    
#            #encoded_words = encoded_words0.permute(0, 2, 1)
#            
#            ### sentence level
#            encoded_sent = self.sent_level(encoded_words0).view(-1, self.max_num_sents, self.num_wordlevel_explainer)
#    
#            #encoded_sent0 = encoded_sent0.squeeze(-1)
#            encoded_sent = self.sent_level2(encoded_sent)
#            #encoded_sent = encoded_sent0.unsqueeze(-2).expand(-1, self.max_num_words, -1)
#    
#            ### concatencation 
#            logits_T = encoded_sent.view(-1, self.max_num_sents)
#    
#            ### sentence level
#            encoded_sent0 = self.sent_level(encoded_words0)
#    
#            encoded_sent0 = encoded_sent0.squeeze(-1)
#            encoded_sent0 = self.sent_level2(encoded_sent0)
#            encoded_sent = encoded_sent0.unsqueeze(-2).expand(-1, self.max_num_words, -1)
#            
#            ### review level information
#            encoded_review = encoded_sent0.view(-1, self.max_num_sents, self.num_sentlevel_explainer)
#            encoded_review = encoded_review.mean(-2)
#            encoded_review = nn.Linear(in_features = self.num_sentlevel_explainer,
#                                       out_features = self.num_reviewlevel_explainer)(encoded_review)
#            
#            ### concatencation 
#            combined = torch.cat((encoded_words0, encoded_sent), -1).view(-1, 1, self.num_wordlevel_explainer + self.num_sentlevel_explainer)
#            logits_T = self.concat_level(combined.squeeze(-1))
#            logits_T = logits_T.view(-1, self.max_num_sents * self.max_num_words)
#            logits_T = self.concat_level2(logits_T)

        return logits_T
        
    def approximater(self, x, Z_hat, num_sample):

        assert num_sample > 0
#        ## Upsampling
#        if self.chunk_size > 1: 
#            
#            Z_hat0 = F.interpolate(Z_hat0, scale_factor = (self.chunk_size, self.chunk_size), mode = 'nearest')
#              
    
        ## sentence encoding
        embeded_words = self.embedding_layer(x).view(-1, self.max_num_sents, self.max_num_words, self.embedding_dim) # sent * word * d
        newsize = [embeded_words.size(0), num_sample]
        newsize.extend(list(map(lambda x: x, embeded_words.size()[1:])))
            
        if self.chunk_size == self.max_num_words:

            Z_hat = Z_hat.unsqueeze(-1).unsqueeze(-1).expand(torch.Size(newsize))
        
        elif self.chunk_size == 1:
            
            Z_hat = Z_hat.view(Z_hat.size(0), Z_hat.size(1), self.max_num_sents, -1).unsqueeze(-1).expand(torch.Size(newsize))
        
        else:
            
            Z_hat = Z_hat.view(Z_hat.size(0), Z_hat.size(1) * self.max_num_sents, -1)
            Z_hat = nn.Sequential(
                    nn.ConstantPad1d(self.chunk_size - 1, 0),
                    nn.MaxPool1d(kernel_size = self.chunk_size, stride = 1, padding = 0)
                    )(Z_hat)
            Z_hat = Z_hat.view(Z_hat.size(0), -1, self.max_num_sents, self.max_num_words).unsqueeze(-1).expand(torch.Size(newsize))

        embeded_words = torch.mul(embeded_words.unsqueeze(1).expand(torch.Size(newsize)), Z_hat)
        
        if self.approximater_type in ['rnn', 'RNN', 'lstm', 'LSTM', 'gru', 'GRU']:
             
            embeded_words = embeded_words.view(-1, self.max_num_words, self.embedding_dim, 1).squeeze(-1)
            embeded_words = self.encoder_word(embeded_words)[0] # batch * sent, word-dim, hidden-dim

            encoded_sent = torch.mean(embeded_words, -2).view(-1, self.max_num_sents, self.num_rnn_out_size * self.mul_bidirect)

            encoded_sent = self.encoder_sent(encoded_sent)[0]
            encoded_review = encoded_sent.contiguous().view(-1, self.max_num_sents * 2 * self.mul_bidirect)
            
            encoded_review = self.dense(encoded_review)

            pred = F.log_softmax(encoded_review, dim = -1)
            
            if num_sample > 1:
                
                pred = pred.view(-1, num_sample, pred.size(-1))
                pred = pred.mean(1)

        elif self.approximater_type in ['lstm-light', 'LSTM-light', 'LSTM-Light', 'LSTM-LIGHT']:
            
            embeded_words = embeded_words.view(-1, self.max_num_sents * self.max_num_words, self.embedding_dim, 1).squeeze(-1)
            embeded_words = self.encoder_word(embeded_words)[0]
            encoded_review = self.encoder_sent(embeded_words.contiguous().view(embeded_words.size(0), -1, 1).squeeze(-1))
            pred = F.log_softmax(encoded_review, dim = -1)
            
            if num_sample > 1:
                
                pred = pred.view(-1, num_sample, pred.size(-1))
                pred = pred.mean(1)
                
        elif self.approximater_type in ['cnn', 'CNN']:

            #print(embeded_words.size())
            encoded_sent = self.encoder_word(embeded_words.view(-1, 1, self.max_num_words * self.embedding_dim, 1).squeeze(-1))
            encoded_sent = encoded_sent.view(-1, self.max_num_sents, self.num_cnn_hidden_size) # batch * 15 * 100
        
            ## feature selection
            if num_sample > 1:
                
                #encoded_sent = encoded_sent.unsqueeze(1)
                #encoded_sent = torch.mul(encoded_sent.expand(encoded_sent.size(0), num_sample, encoded_sent.size(-2), encoded_sent.size(-1)), Z_hat.unsqueeze(-1))
                ## decoder
                encoded_review = torch.mean(encoded_sent, -2) # [10, n, 250]
                pred = self.encoder_sent(encoded_review.view(-1,encoded_review.size(-1)))
                pred = pred.view(-1, num_sample, pred.size(-1))
                pred = pred.mean(1) # shoudl be [10, n, 2]#?check whether it has differnet value...
                
            elif num_sample == 1:

                #print(encoded_sent.size(), Z_hat.size())
                
                #encoded_sent = torch.mul(encoded_sent, Z_hat.unsqueeze(-1)) # should be [10, 15, 1] * [10, 15, 250] -> [10, 15, 250]  
                ## decoder
                encoded_review = torch.mean(encoded_sent, -2) # batch * 100
                pred = self.encoder_sent(encoded_review) # shoudl be [batch, 2]
                
            else:
                raise ValueError('num_sample should be a positive integer') 
            
        else:
            raise UnknownModelError()
    
        return pred
    
    def reparameterize(self, p_i, tau, k, num_sample = 1):
        '''
        2018 (c) Seojin Bang
        Sampled from Gumbel-softmax distribution. Reference: L2X
        '''

        ## sampling
        p_i_ = p_i.view(p_i.size(0), 1, 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, k, p_i_.size(-1))
        C_dist = RelaxedOneHotCategorical(tau, p_i_)
        V = torch.max(C_dist.sample(), -2)[0] # [batch-size, multi-shot, d]

        ## without sampling
        V_fixed_size = p_i.unsqueeze(1).size()
        _, V_fixed_idx = p_i.unsqueeze(1).topk(k, dim = -1) # batch * 1 * k
        V_fixed = idxtobool(V_fixed_idx, V_fixed_size, is_cuda = self.args.cuda)
        V_fixed = V_fixed.type(torch.float)

        return V, V_fixed 
        
    def forward(self, x, num_sample = 1):

        p_i = self.explainer(x) # probability of each element to be selected [barch-size, d]
        Z_hat, Z_hat_fixed = self.reparameterize(p_i, tau = self.tau, k = self.K, num_sample = num_sample) # torch.Size([batch-size, num-samples for multishot prediction, d]), d-dimentional random vector to approximate k-hot random explainer Z during training.
        #print(x.size(), Z_hat.size(), num_sample)
        logit = self.approximater(x, Z_hat, num_sample)
        logit_fixed = self.approximater(x, Z_hat_fixed, num_sample = 1)

        return logit, p_i, Z_hat, logit_fixed

    def weight_init(self):
        xavier_init(self._modules)      
#        
#    def weight_init(self):
#        for m in self._modules:
#            xavier_init(self._modules[m])                
                
def prior(var_size):
    '''
    2018 (c) Seojin Bang
    prior distribution as Gumbel-softmax distribution.
    '''
    p = torch.ones(var_size[1])/ var_size[1]
    p = p.view(1, var_size[1])
    p_prior = p.expand(var_size) # [batch-size, k, feature dim]
    
    return p_prior
#%%
def xavier_init(ms):
    
    for m in ms :
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean = 1, std = 0.02)
            init.constant(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean = 1, std = 0.02)
            init.constant(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean = 1, std = 0.02)
            init.constant(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal(param.data)
                else:
                    init.normal_(param.data)

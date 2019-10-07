import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from utils import Flatten, idxtobool, cuda

class Explainer(nn.Module):

    def __init__(self, **kwargs):
        
        super(Explainer, self).__init__()

        self.args = kwargs['args']
        self.mode = self.args.mode

        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.tau = 0.5 # float, parameter for concrete random variable distribution
        self.K = self.args.K # number of variables selected
        self.model_type = self.args.explainer_type# 'nn', 'cnn', 'nn4', 'cnn4'
        self.chunk_size = self.args.chunk_size

        self.encode_cnn = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size = (5, 5), padding = 2),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size = (2, 2)),
                nn.Conv2d(4, 1, kernel_size = (5, 5), padding = 2),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size = (2, 2)),
                #nn.Conv2d(16, 1, kernel_size = (1, 1)),
                Flatten(),
                nn.Linear(49, 784),
                nn.LogSoftmax(1)
                )

        self.encode_cnn2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 5), padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2)), #torch.Size([50, 8, 14, 14])
            nn.Conv2d(8, 1, kernel_size=(5, 5), padding=2), # torch.Size([50, 16, 14, 14])
            #nn.ReLU(True),
            #nn.MaxPool2d(kernel_size=(2, 2)), # torch.Size([50, 16, 7, 7])
            #nn.Conv2d(16, 1, kernel_size=(1, 1)),# torch.Size([50, 1, 7, 7])
            Flatten(),
            nn.LogSoftmax(1)
        )
        
        self.encode_cnn4 = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size = (5, 5), padding = 2),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size = (2, 2)),
                nn.Conv2d(8, 16, kernel_size = (5, 5), padding = 2),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size = (2, 2)),
                nn.Conv2d(16, 1, kernel_size = (1, 1)),
                Flatten(),
                nn.LogSoftmax(1)
                )
        
        self.decode_cnn = nn.Sequential(
                 nn.Conv2d(1, 32, kernel_size = (5, 5), padding = 2),
                 nn.ReLU(True),
                 nn.MaxPool2d(kernel_size = (2, 2)),
                 nn.Conv2d(32, 64, kernel_size = (5, 5), padding = 2),
                 nn.ReLU(True),
                 nn.MaxPool2d(kernel_size = (2, 2)),
                 Flatten(),
                 nn.Linear(7 * 7 * 64, 10),
                 nn.LogSoftmax(1)
                 )
        
        self.encode_nn = nn.Sequential(
            Flatten(),
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.LogSoftmax(1)
            )

        self.decode_nn = nn.Sequential(
                Flatten(),
                nn.Linear(784, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 10),
                nn.LogSoftmax(1)
                )
        
        self.encode_nn4 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 49),
            nn.LogSoftmax(1)
            )
        
        if self.model_type == 'nn': 
            
            self.encode = self.encode_nn
            self.decode = self.decode_nn

        elif self.model_type == 'nn4':
        
            self.encode = self.encode_nn4
            self.decode = self.decode_nn           
        
        elif self.model_type == 'cnn':
        
            self.encode = self.encode_cnn
            self.decode = self.decode_cnn
        
        elif self.model_type == 'cnn4':

            self.encode = self.encode_cnn4
            self.decode = self.decode_cnn

        elif self.model_type == 'cnn2':

            self.encode = self.encode_cnn2
            self.decode = self.decode_cnn

        else:
        
            ValueError('invalid model_type') 
    
    def encoder(self, x):
        
        p_i = self.encode(x)
        
        return p_i #[batch_size, d]
        
    def decoder(self, x, Z_hat, num_sample = 1):
        
        assert num_sample > 0

        Z_hat0 = Z_hat.view(Z_hat.size(0), Z_hat.size(1),
                            int(np.sqrt(Z_hat.size(-1))),
                            int(np.sqrt(Z_hat.size(-1))))

        ## Upsampling
        if self.chunk_size > 1: 

            Z_hat0 = F.interpolate(Z_hat0,
                                   scale_factor = (self.chunk_size, self.chunk_size),
                                   mode = 'nearest')

        ## feature selection
        newsize = [x.size(0), num_sample]
        newsize.extend(list(map(lambda x: x, x.size()[2:])))
        net = torch.mul(x.expand(torch.Size(newsize)), Z_hat0) # torch.Size([batch_size, num_sample, d])
        
        ## decode
        newsize2 = [-1, 1]
        newsize2.extend(newsize[2:])
        net = net.view(torch.Size(newsize2)) 
        pred = self.decode(net)
        pred = pred.view(-1, num_sample, pred.size(-1))
        pred = pred.mean(1)
        
        return pred
    
    def reparameterize(self, p_i, tau, k, num_sample = 1):

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

        p_i = self.encoder(x) # probability of each element to be selected [barch-size, d]
        Z_hat, Z_hat_fixed = self.reparameterize(p_i, tau = self.tau,
                                                 k = self.K,
                                                 num_sample = num_sample) # torch.Size([batch-size, num-samples, d])
        logit = self.decoder(x, Z_hat, num_sample)
        logit_fixed = self.decoder(x, Z_hat_fixed)

        return logit, p_i, Z_hat, logit_fixed
        
    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

def prior(var_size):

    p = torch.ones(var_size[1])/ var_size[1]
    p = p.view(1, var_size[1])
    p_prior = p.expand(var_size) # [batch-size, k, feature dim]
    
    return p_prior

def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()

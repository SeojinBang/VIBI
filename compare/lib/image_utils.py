#import cv2
import numpy as np
import torch
import copy
from torch.autograd import Variable
import sys
sys.path.append('../')
from utils import index_transfer, cuda


def preprocess_image(img, cuda=False):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    if cuda:
        preprocessed_img = Variable(preprocessed_img.cuda(), requires_grad=True)
    else:
        preprocessed_img = Variable(preprocessed_img, requires_grad=True)

    return preprocessed_img

class Segmentation(object):

    def __init__(self, dataset, filter_size = (4, 4), is_cuda = False): #change

        self.filter_size = filter_size
        self.is_cuda = is_cuda
        self.dataset = dataset
        #self.img = img
        #func = getattr(self, dataset)
        #self.output = func()
    
    def __call__(self, img):
        self.img = cuda(torch.tensor(img.squeeze(0), dtype = torch.float), self.is_cuda)
        func = getattr(self, self.dataset)
        return func(self.img)
        
    def default(self, img):
        
        original_ncol = self.img.size(-1) 
        original_nrow = self.img.size(-2)
        num = original_ncol * original_nrow // (self.filter_size[0] * self.filter_size[1])
        seg = torch.clone(self.img)
        for idx in range(num):
            idx = cuda(torch.tensor([idx], dtype = torch.long), self.is_cuda)
            idx_all = index_transfer(dataset = 'mnist', 
                                     idx = idx, 
                                     filter_size = self.filter_size, 
                                     original_ncol = original_ncol, 
                                     original_nrow = original_nrow, 
                                     is_cuda = self.is_cuda).output.squeeze(0) 
            seg.view(1, -1)[:,idx_all] = cuda(torch.tensor(idx, dtype = torch.float), self.is_cuda)
        
        seg = seg.detach().cpu().numpy().astype(int)
        
        return seg
        
    def mnist(self, img):    
        
        return self.default(img) 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:01:33 2018

@author: seojin.bang
"""
import argparse
import numpy as np
import re

def main(args):
    
    dataset = args.dataset
    blackbox = args.blackbox
    method = args.method
    result_dir = args.result_dir
    
    measures = ['avg_acc', 'avg_acc_fixed', 'acc_zeropadded', 
                'vmi', 'vmi_fixed', 'vmi_zeropadded', 
                'precision_macro_approx', 'precision_macro_approx_fixed', 'precision_macro_zeropadded', 
                'recall_micro_approx', 'recall_micro_approx_fixed', 'recall_micro_zeropadded', 
                'f1_macro_approx', 'f1_macro_approx_fixed', 'f1_macro_zeropadded', 
                'precision_micro_approx', 'precision_micro_approx_fixed', 'precision_micro_zeropadded', 
                'recall_macro_approx', 'recall_macro_approx_fixed', 'recall_macro_zeropadded',
                'f1_micro_approx', 'f1_micro_approx_fixed', 'f1_micro_zeropadded']

    measures_order = ['acc_zeropadded', 'avg_acc', 'avg_acc_fixed', 
                'precision_macro_zeropadded', 'precision_macro_approx', 'precision_macro_approx_fixed', 
                'precision_micro_zeropadded', 'precision_micro_approx', 'precision_micro_approx_fixed', 
                'recall_macro_zeropadded', 'recall_macro_approx', 'recall_macro_approx_fixed',
                'recall_micro_zeropadded', 'recall_micro_approx', 'recall_micro_approx_fixed',
                'f1_macro_zeropadded', 'f1_macro_approx', 'f1_macro_approx_fixed',
                'f1_micro_zeropadded', 'f1_micro_approx', 'f1_micro_approx_fixed',
                'vmi', 'vmi_fixed', 'vmi_zeropadded']
    
    if dataset is 'mnist':
        K = [64, 96, 160, 320, 16, 24, 40, 80, 4, 6, 10, 20]
        chunksize = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4]
        assert len(K) == len(chunksize)
    
        fns = np.array([])
        for i in range(len(K)):
            fns = np.append(fns, ['log_' + dataset + '_' + method + '_K' + str(K[i]) + '_chunk' + str(chunksize[i]) + '.txt'])
            
    elif dataset is 'imdb':
        K = [1, 5, 15, 1, 3, 5, 10, 15]
        chunksize = [50, 1, 1, 5, 5, 5, 5, 5]
        assert len(K) == len(chunksize)
        
        fns = np.array([])
        for i in range(len(K)):
            fns = np.append(fns, ['log_' + dataset + '_' + blackbox + '_' + method + '_K' + str(K[i]) + '_chunk' + str(chunksize[i]) + '.txt'])
    else:
        raise Exception('unknown dataset')
    
    for mea_idx in range(len(measures)):
        for fn in fns:
            f = open(Path(result_dir).joinpath(fn), 'r')
            if f.mode == 'r':
                contents = f.read()
                idx = contents.find(measures[mea_idx])
                if measures[mea_idx] is not measures_order[len(measures_order)-1]:
                    measures_next = measures_order[measures_order.index(measures[mea_idx]) + 1]
                    idx_next = contents.find(measures_next)
                else:
                    idx_next = len(contents)
                val = contents[(idx + len(measures[mea_idx]) + 1):(idx_next - 1)]
                val = re.sub('[a-zA-Z\n]', '', val)
                print(val)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default = 'imdb', type = str, help='dataset name: imdb-sent, imdb-word, mnist, mimic')
    parser.add_argument('--result_dir', default = './result', type = str, help='Result directory path')
    parser.add_argument('--blackbox', default = 'lstm-light-onedirect', type = str, help='blackbox model type')
    parser.add_argument('--method', default = 'saliency', type = str, help='interpretable learning method')

    args = parser.parse_args()
    
    main(args)

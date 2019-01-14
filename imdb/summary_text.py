#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:01:33 2018

@author: seojin.bang
"""
import argparse
import numpy as np
import re
import csv

def main(args):
    
    filename = args.filename
    result_dir = args.result_dir

    print('summarize {} and save in {}'.format(filename, result_dir))
    
    fns = np.array([filename + '_idx0.txt', filename + '_idx200.txt'])

    contents_all = []
    sent = []
    pred = []
    approx = []
    for fn in fns:
        #f = open(Path(result_dir).joinpath(fn), 'r')
        f = open(result_dir + '/' + fn, 'r')
        if f.mode == 'r':
            contents = f.read()
            contents = re.sub('_+', ' ', contents).strip()
            contents = re.sub('\s{2,}', ', ', contents)
            contents = contents.split("sent:")
            contents = list(filter(None, contents))
            contents = ["sent:" + x.strip() for x in contents]
            contents = [item[(item.find('approx:')+10):] for item in contents]
            contents = [re.sub(r'^,|,$', "", item).strip() for item in contents]

            contents_all = contents_all.extend(contents)
            sent = sent.extend([item[(item.find('sent:')+5):(item.find('sent:')+8)] for item in contents])
            pred = pred.extend([item[(item.find('pred:')+5):(item.find('pred:')+8)] for item in contents])
            approx = approx.extend([item[(item.find('approx:')+7):(item.find('approx:')+10)] for item in contents])            
            
    mydata = [contents_all, sent, pred, approx]
    myfile = open(filename + '.csv', 'w')
    with myfile:
        writer = csv.writer(myfile)
        writer.writerows(mydata)

    print(mydata)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', default = 'None', type = str, help='File name')
    parser.add_argument('--result_dir', default = './result', type = str, help='Result directory path')

    args = parser.parse_args()
    
    main(args)

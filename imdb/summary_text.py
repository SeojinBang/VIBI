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
    output_dir = args.output_dir
    
    print('summarize {} and save in {}'.format(filename, result_dir))
    
#    fns = np.array([filename + '_idx0.txt', filename + '_idx200.txt'])
    idx_list = [0, 1, 2, 4, 120, 121, 123, 124, 197, 198, 199, 200]
    fns = np.array([filename + '_idx' + str(item) + '.txt' for item in idx_list])
    
    contents_all = []
    sent = []
    pred = []
    approx = []
    for fn in fns:
        #f = open(Path(result_dir).joinpath(fn), 'r')
        f = open(result_dir + '/' + fn, 'r')
        if f.mode == 'r':
            contents = f.read()
            contents = re.sub(' br ', '...<br />...', contents)
            contents = re.sub('<unk>', '...', contents)
            contents = re.sub('_+', ' ', contents).strip()
            contents = re.sub('\s{2,}', '...<br />... ', contents)
            contents = contents.split("sent:")
            contents = list(filter(None, contents))
            contents = ["sent:" + x.strip() for x in contents]

            sent.extend([item[(item.find('sent:')+5):(item.find('sent:')+8)] for item in contents])
            pred.extend([item[(item.find('pred:')+5):(item.find('pred:')+8)] for item in contents])
            approx.extend([item[(item.find('approx:')+7):(item.find('approx:')+10)] for item in contents])            

            contents = [item[(item.find('approx:')+10):] for item in contents]
            contents = [re.sub(r'^<br />|<br />$', '', item.strip()).strip() for item in contents]

            contents_all.extend(contents)
        f.close()

    
    mydata = zip(contents_all, sent, pred, approx)
    myfile = open(output_dir + '/' + filename + '.csv', 'w')
    with myfile:
        writer = csv.writer(myfile)
        writer.writerow(('keys', 'sent', 'pred', 'approx'))
        for row in mydata:
            if row[0] != '' and row[0] is not None:
                writer.writerow(row)

    print(mydata)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filename', type = str, help='File name')
    parser.add_argument('--result_dir', type = str, help='Result directory path')
    parser.add_argument('--output_dir', default = '.', type = str, help='save output in')
    args = parser.parse_args()
    
    main(args)

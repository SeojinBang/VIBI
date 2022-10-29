#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import io
import re
import torch
import shutil
import requests
import tarfile
import numpy as np
from torchtext import data
from nltk import tokenize
from keras.preprocessing.text import text_to_word_sequence

class IMDB_modified(data.Dataset):

    urls = ['http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz']
    name = 'imdb'
    dirname = 'aclImdb'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path,
                 text_field, label_field, label_pred_field,
                 fname_field, model_name,
                 root_path, load_pred = False, **kwargs):
        """Create an IMDB dataset instance given a path and fields.
        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field),
                  ('label', label_field),
                  ('label_pred', label_pred_field),
                  ('fname', fname_field)]
        examples = []

        ## load predicted label
        model_name_base = os.path.splitext(model_name)[0]
        file_name = model_name_base + '_pred_' + os.path.basename(path) + '.pt'
        path_to_file = os.path.join(root_path, 'processed', file_name)
        
        if os.path.exists(path_to_file): 

            print('...loading the labels predicted by', model_name_base)
            label_pred_dict = torch.load(path_to_file)
           
        else:
            if load_pred:
                raise RuntimeError('No label predicted by {} exists'.format(model_name_base))
                
            model_name = None
        
        ## load original data
        for label in ['pos', 'neg']:
            label_num = {'pos': '3', 'neg': '2'}
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                fname_sub = int(label_num[label] + os.path.basename(fname).split("_")[0])
                with io.open(fname, 'r', encoding = "utf-8") as f:
                    text = f.readline()
                    label_pred = [99] if model_name is None else [label_pred_dict[fname_sub]]
                examples.append(data.Example.fromlist([text, label, label_pred, [fname_sub]],
                                                      fields))

        super(IMDB_modified, self).__init__(examples, fields, **kwargs)     
    
    @classmethod
    def splits(cls,
               text_field, label_field, label_pred_field,
               fname_field, root ='.data',
               train='train', test='test', validation = 'valid', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        #print('imdb-modified-splits')

        return super(IMDB_modified, cls).splits(root = root, text_field = text_field, 
                    label_field = label_field, label_pred_field = label_pred_field,
                    fname_field = fname_field, 
                    train=train, validation=validation, test=test,
                    root_path = root, **kwargs)

    @classmethod
    def iters(cls, batch_size = 32, device = -1, root = '.data', vectors = None, **kwargs):
        """Creater iterator objects for splits of the IMDB dataset.
        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the imdb dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)
            Remaining keyword arguments: Passed to the splits method.
        """
        #print('imdb-modified-iters')
        TEXT = data.Field()
        LABEL = data.Field(sequential = False)
        LABEL_PRED = data.Field(sequential = False)
        FNAME = data.Field(sequential = False)
        
        train, test, valid = cls.splits(TEXT, LABEL, LABEL_PRED, FNAME,
                                        root=root, **kwargs)
        
        TEXT.build_vocab(train, vectors = vectors)
        LABEL.build_vocab(train)
        LABEL_PRED.build_vocab(train)
        FNAME.build_vocab(train)
        
        return data.Iterator.splits(
            (train, test, valid), batch_size=batch_size, device=device) 
    
    @classmethod
    def download(cls, root, check = None):
        """Download and unzip an online archive (.zip, .gz, or .tgz).
        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.
        Returns:
            str: Path to extracted dataset.
        """
        path = os.path.join(root, cls.name)
        check = path if check is None else check
        
        if not os.path.isdir(check):

            for url in cls.urls:
                
                if isinstance(url, tuple):
                    url, filename = url
                else:
                    filename = os.path.basename(url)
                zpath = os.path.join(path, filename)

                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print('downloading {}'.format(filename))
                    download_from_url(url, zpath)
                zroot, ext = os.path.splitext(zpath)
                _, ext_inner = os.path.splitext(zroot)
                
                with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        def is_within_directory(directory, target):
                            
                            abs_directory = os.path.abspath(directory)
                            abs_target = os.path.abspath(target)
                        
                            prefix = os.path.commonprefix([abs_directory, abs_target])
                            
                            return prefix == abs_directory
                        
                        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        
                            for member in tar.getmembers():
                                member_path = os.path.join(path, member.name)
                                if not is_within_directory(path, member_path):
                                    raise Exception("Attempted Path Traversal in Tar File")
                        
                            tar.extractall(path, members, numeric_owner=numeric_owner) 
                            
                        
                        safe_extract(tar, path=path, members=dirs)

                ## separate validation from test set
                print("divide the original test set into (new) test and validation sets")
                np.random.seed(0)
                for label in ['pos', 'neg']:
                    
                    source_from = os.path.join(path, 'aclImdb', 'test', label)
                    source_to = os.path.join(path, 'aclImdb', 'valid', label)

                    if not os.path.exists(source_to):
                        os.makedirs(source_to)

                    num_files = len(os.listdir(source_from))
                    indices = np.arange(num_files)
                    np.random.shuffle(indices)
                    indices = indices[range(num_files // 2)]

                    for file_idx in indices:

                        file_name = glob.glob('{}/{}_*.txt'.format(str(source_from), file_idx))
                        assert len(file_name) == 1
                        shutil.move(str(file_name[0]), source_to)

                    np.random.seed(1)

            print("downloaded!")

        return os.path.join(path, cls.dirname)

def tokenizer_twolevel(comment):
    
    max_num_words = 50#100
    max_num_sents = 15
    pad_token = '<pad>'
    
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", 
        str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    comment = comment.strip().lower()
    
    ## sentence tokenize    
    comment = tokenize.sent_tokenize(comment)[:max_num_sents]
    
    ## word tokenize
    text = []
    for idx in range(max_num_sents):
        
        if idx < len(comment):
            
            sent = comment[idx]
            sent = text_to_word_sequence(sent)[:max_num_words]
            num_pads_add = max_num_words - len(sent) 
            sent.extend([pad_token] * num_pads_add)
            text.extend(sent) 
            
        else:
            
            sent = [pad_token] * max_num_words
            text.extend(sent)
    
    return text

def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

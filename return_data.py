import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from mnist.data_loader import MNIST_modified
from torchtext import data
from torchtext.vocab import GloVe
from imdb.data_loader import IMDB_modified, tokenizer_twolevel

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def return_data(args):

    name = args.dataset
    root = args.root
    batch_size = args.batch_size
    data_loader = dict()
    device = 0 if args.cuda else -1

    if name in ['mnist', 'MNIST']:

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),])

        train_kwargs = {'root':root, 'mode':'train', 'transform':transform, 'download':True, 
                        'load_pred': args.load_pred, 'model_name': args.model_name}
        valid_kwargs = {'root':root, 'mode':'valid', 'transform':transform, 'download':True, 
                        'load_pred': args.load_pred, 'model_name': args.model_name}
        test_kwargs = {'root':root, 'mode':'test', 'transform':transform, 'download':False, 
                       'load_pred': args.load_pred, 'model_name': args.model_name}
        dset = MNIST_modified

        train_data = dset(**train_kwargs)
        valid_data = dset(**valid_kwargs)
        test_data = dset(**test_kwargs)

        # data loader
        num_workers = 0
        train_loader = DataLoader(train_data,
                                   batch_size = batch_size,
                                   shuffle = True,
                                   num_workers = num_workers,
                                   drop_last = True,
                                   pin_memory = True)

        valid_loader = DataLoader(valid_data,
                                   batch_size = batch_size,
                                   shuffle = False,
                                   num_workers = num_workers,
                                   drop_last = False,
                                   pin_memory = True)
        
        test_loader = DataLoader(test_data,
                                    batch_size = batch_size,
                                    shuffle = False,
                                    num_workers = num_workers,
                                    drop_last = False,
                                    pin_memory = True)
    
        data_loader['x_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        data_loader['y_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        
    elif name in ['imdb', 'IMDB']:

        embedding_dim = 100
        max_total_num_words = 20000
        text = data.Field(tokenize = tokenizer_twolevel, 
                          batch_first = True)
        label = data.Field(lower = True)
        label_pred = data.Field(use_vocab = False, fix_length = 1)
        fname = data.Field(use_vocab = False, fix_length = 1)
        
        train, valid, test = IMDB_modified.splits(text, label, label_pred, fname,
                                                  root = root, model_name = args.model_name,
                                                  load_pred = args.load_pred)
        print("build vocab...")
        text.build_vocab(train, vectors = GloVe(name = '6B',
                                                dim = embedding_dim,
                                                cache = root), max_size = max_total_num_words)
        label.build_vocab(train)
        
        print("Create Iterator objects for multiple splits of a dataset...")
        train_loader, valid_loader, test_loader = data.Iterator.splits((train, valid, test),
                                                                       batch_size = batch_size,
                                                                       device = device,
                                                                       repeat = False)
        
        data_loader['word_idx'] = text.vocab.itos
        data_loader['x_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        data_loader['y_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        data_loader['max_total_num_words'] = max_total_num_words
        data_loader['embedding_dim'] = embedding_dim
        data_loader['max_num_words'] = 50
        data_loader['max_num_sents'] = int(next(iter(train_loader)).text.size(-1) / data_loader['max_num_words'])

    else : raise UnknownDatasetError()
    
    data_loader['train'] = train_loader
    data_loader['valid'] = valid_loader
    data_loader['test'] = test_loader

    return data_loader

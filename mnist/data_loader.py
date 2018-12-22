from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os.path
import errno
import numpy as np
import torch
import codecs
from torchvision.datasets.utils import download_url

class MNIST_modified(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    valid_file = 'valid.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):

        if self.mode is 'train':
            return self.train_labels
        elif self.mode is 'valid':
            return self.valid_labels      
        else:
            return self.test_labels

    def __init__(self, root, mode = 'train', transform = None, target_transform = None, download = False, load_pred = False, model_name = 'original'):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode  # training set or test set
        model_name = os.path.splitext(os.path.basename(model_name))[0]
        self.training_pred_file = model_name + '_pred_train.pt'
        self.valid_pred_file = model_name + '_pred_test.pt'
        self.test_pred_file = model_name + '_pred_valid.pt'
        
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.mode is 'train':
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.train_predlabels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_pred_file)) if load_pred else self.train_labels
        elif self.mode is 'valid':
            self.valid_data, self.valid_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.valid_file))    
            self.valid_predlabels = torch.load(
                os.path.join(self.root, self.processed_folder, self.valid_pred_file)) if load_pred else self.valid_labels
        elif self.mode is 'test':
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_predlabels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_pred_file)) if load_pred else self.test_labels
        else:
            raise RuntimeError("mode error. 'mode' should be either train, valid, or test")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        if self.mode is 'train':
            img, target, pred = self.train_data[index], self.train_labels[index], self.train_predlabels[index] 
        elif self.mode is 'valid':
            img, target, pred = self.valid_data[index], self.valid_labels[index], self.valid_predlabels[index]    
        elif self.mode is 'test':
            img, target, pred = self.test_data[index], self.test_labels[index], self.test_predlabels[index]
        else:
            raise RuntimeError("mode error. 'mode' should be either train, valid, or test")

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, pred, index

    def __len__(self):
        
        if self.mode is 'train':
            return len(self.train_data)
        elif self.mode is 'valid':
            return len(self.valid_data)   
        elif self.mode is 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("mode error. 'mode' should be either train, valid, or test")
            

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.valid_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        #from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            download_url(url, root=os.path.join(self.root, self.raw_folder),
                         filename=filename, md5=None)
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_valid_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        
        # Define the indices
        indices = list(range(len(training_valid_set[0])))
        if os.path.exists(os.path.join(self.root, self.processed_folder, 'valid_idx.npy')):
            valid_idx = np.load(os.path.join(self.root, self.processed_folder, 'valid_idx.npy'))
        else: 
            valid_idx = np.random.choice(indices, size = 10000, replace = False)
            np.save(os.path.join(self.root, self.processed_folder, 'valid_idx.npy'), valid_idx)
        train_idx = list(set(indices) - set(valid_idx))
        
        training_set = (
            training_valid_set[0][train_idx],
            training_valid_set[1][train_idx] 
        )
        valid_set = (
            training_valid_set[0][valid_idx],
            training_valid_set[1][valid_idx] 
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.valid_file), 'wb') as f:
            torch.save(valid_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        #images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
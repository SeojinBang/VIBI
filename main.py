import numpy as np
import torch
import argparse
from utils import str2bool
from solver import Solver

def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ## print-option
    np.set_printoptions(precision=4) # upto 4th digits for floating point output
    torch.set_printoptions(precision=4)
    print('\n[ARGUMENTS]\n', args)

    ## cuda
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda True")
    args.cuda = (args.cuda and torch.cuda.is_available())
    
    net = Solver(args)

    if args.mode == 'train' :
        net.train(test = False)
    elif args.mode == 'test' :
        net.train(test = True)
        #net.val(test = True)
    else : 
        print('\nError: "--mode train" or "--mode test" expected')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'VIBI for interpretation')
    parser.add_argument('--epoch', default = 100, type = int, help = 'epoch number')
    parser.add_argument('--lr', default = 1e-4, type = float, help = 'learning rate')
    parser.add_argument('--beta', default = 0.1, type = float, help = 'beta for balance between information loss and prediction loss')
    parser.add_argument('--K', required = True, type = int, help='dimension of encoding Z')
    parser.add_argument('--chunk_size', default = -1, type = int, help='chunk size. for image, chunk x chunk will be the actual chunk size')
    parser.add_argument('--num_avg', default = 12, type = int, help='the number of samples when perform multi-shot prediction')
    parser.add_argument('--batch_size', default = 50, type = int, help='batch size')
    parser.add_argument('--env_name', default = 'main', type = str, help='visdom env name')
    parser.add_argument('--dataset', required = True, type = str, help='dataset name: imdb, mnist')
    parser.add_argument('--model_name', required = True, type = str, help='model names to be interpreted')
    parser.add_argument('--explainer_type', default = 'None', type = str, help='explainer types: nn, cnn for mnist')    
    parser.add_argument('--approximater_type', default = 'None', type = str, help='explainer types: nn, cnn')    
    parser.add_argument('--load_checkpoint',default = '', type = str, help = 'checkpoint name')
    parser.add_argument('--checkpoint_name',default = 'best_acc.tar', type = str, help = 'checkpoint name')
    parser.add_argument('--default_dir', default = '.', type = str, help='default directory path')
    parser.add_argument('--data_dir', default = 'dataset', type = str, help='data directory path')
    parser.add_argument('--summary_dir', default = 'summary', type = str, help='summary directory path')
    parser.add_argument('--checkpoint_dir', default = 'checkpoints', type = str, help='checkpoint directory path')
    parser.add_argument('--cuda', default = True, type = str2bool, help = 'enable cuda')
    parser.add_argument('--mode', default = 'train', type=str, help = 'train or test')
    parser.add_argument('--tensorboard',default = True, type= str2bool, help='enable tensorboard')
    parser.add_argument('--save_image',default = True, type= str2bool, help='if True, then save images')
    parser.add_argument('--save_checkpoint',default = False, type= str2bool, help='if True, then save checkpoint')
    parser.add_argument('--tau', default = 0.7, type = float, help = 'tau')
    args = parser.parse_args()

    main(args)


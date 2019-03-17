import sys
import pdb
from math import floor
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np 
from os.path import join
from time import time
from torch.utils import data
sys.path.append('..')

from models.shallow_cnn.shallow_cnn import ShallowCNN
from datasets.datasets import RobotNavDataset, SubmissionDataset
from utils.training import EarlyStopping, train, test
from utils.logging import setup_logger
from utils.submissions import make_submission

DATA_DIR = '../data'
SEED = 1337

def main():
    parser = argparse.ArgumentParser(description='Shallow-CNN Training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default:64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default:1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train for (default:100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
                        help='learning rate for optimizer (default:0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--early-stopping', type=int, default=10, metavar='N',
                        help='Patience for early stopping (default:10)')
    parser.add_argument('--data-dir', type=str, default='../data', metavar='path/to/dir',
                        help='path to directory containing data (default:../data)')
    parser.add_argument('--train-size', type=float, default=0.85, metavar='pct',
                        help='fraction of dataset to use for training (default:0.85)')
    parser.add_argument('--test-size', type=float, default=0.15, metavar='pct',
                        help='fraction of dataset to use for testing (default:0.15)')
    parser.add_argument('--dropout-rate', type=float, default=0.5, metavar='pct',
                        help='dropout rate after convolution layers (default:0.5)')
    parser.add_argument('--conv1-width', type=int, default=10, metavar='w',
                        help='Width of 1st convolution kernel (default:10)')
    parser.add_argument('--n_channels', type=int, default=30, metavar='N',
                        help='Number of channels ouput by convolution layers (default:30)')
    parser.add_argument('--max-pool-kernel-size', type=int, default=25, metavar='w',
                        help='Width of max-pool kernel after convolution (default:25)')
    parser.add_argument('--max-pool-stride', type=int, default=5, metavar='N',
                        help='stride along 2nd axis for max-pool (default:5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', metavar='path/to/file',
                        help='file to save checkpoints (default:checkpoint.pt)')


    #TODO add arg to save everything to specific folder

    # Time id used for saving files
    time_id = int(time())

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(SEED)

    # Load the datsets
    print('loading datasets')
    train_set = RobotNavDataset(args.data_dir)
    submission_set = SubmissionDataset(args.data_dir)
    train_size = floor(0.8*len(train_set))
    test_size = floor(0.2*len(train_set))
    train_subset, test_subset = data.random_split(train_set, (train_size, test_size))
    train_loader = torch.utils.data.DataLoader(train_subset,
                                          batch_size=args.batch_size,
                                          shuffle=True)
    # Don't think we actually need shuffle here...
    test_loader = torch.utils.data.DataLoader(test_subset,
                                            batch_size=args.test_batch_size)

    # Initialize objects
    print('creating model')
    model = ShallowCNN(n_channels=args.n_channels,
                        conv1_width=args.conv1_width,
                        max_pool_kernel_size=args.max_pool_kernel_size,
                        max_pool_stride=args.max_pool_stride,
                        dropout_rate=args.dropout_rate)
    model.double() # TODO: look into if this is actually needed...
    early_stopper = EarlyStopping(patience=args.early_stopping, check_file=args.checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logfile = '{}.log'.format(time_id)
    logger = setup_logger(logfile=logfile, console_out=True)
    loss_func = F.nll_loss

    # Train the model
    print('training model')
    for epoch in range(1, args.epochs+1):
        train(model, train_loader, optimizer, loss_func, epoch, log_interval=args.log_interval, log_func=logger.info)
        test_loss = test(model, test_loader, loss_func, log_func=logger.info)
        # Early stopper will handle saving the checkpoints
        if early_stopper(test_loss, model):
            break

    print('creating submission')
    make_submission(model, submission_set.data, 'submission-{}.csv'.format(time_id))




if __name__ == "__main__":
    main()
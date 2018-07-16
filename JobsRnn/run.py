import numpy as np
import argparse
import torch
from .run_experiment import run_experiment
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='seq2seq name cleansing project!')
"""
Experiment Parameters
exp-name is required parameter but there is an opened bug in python that state that required parameters
still appear in optional arguments in the help menu
https://bugs.python.org/issue9694
"""

# """
# Experiment Parameters
# """
parser.add_argument('--exp-name', type=str, default='dummy', help='experiment name to used across everything',  required=True)
parser.add_argument('--random-seed', type=int, default=120, help='random seed')
# # parser.add_argument('--num-gpus', type=int, default=1, help='number of GPUs to be used')
parser.add_argument('--num-workers', type=int, default=2, help='number of workers to be used')
parser.add_argument('--cpu-only', action='store_true', default=False, help='use cpu only to train')
# parser.add_argument('--experimental', action='store_true', default=False, help='enable experimental setup')
# parser.add_argument('--data-sample', type=int, default=1000, help='data sample size; iff experimental setup')
# parser.add_argument('--print-freq', '-p', default=100, type=int, help='print frequency')
# parser.add_argument('--save-freq', default=100, type=int, help='save weights frequency')
# parser.add_argument('--manual-test-freq', default=500, type=int, help='test frequency against manual provided sample')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
# parser.add_argument('--resume-type', type=str,  help='load from best/last checkpoint')
# parser.add_argument('--initial-weights-exp-name', type=str,  help='load weights from specific experiment name')
parser.add_argument('--gpu-device-id', type=str, default='0', help='run on specific gpu device id')

""" Input Parameters """
package_root = "JobsRnnLocal/"
# parser.add_argument('--submission-dir', default=package_root + 'results/', type=str, help='submission dir')
parser.add_argument('--input-dir', type=str, default=package_root + 'input/', help='root input dir')
# parser.add_argument('--train-dir', type=str, default=package_root + 'train/', help='train root  dir')
# parser.add_argument('--test-dir', type=str, default=package_root + 'test/', help='test root dir')
parser.add_argument('--train-csv', type=str, default='train_21M.csv', help='train csv file')
# parser.add_argument('--test-csv', type=str, default='test.csv', help='test csv file')
# parser.add_argument('--manual-test-csv', type=str, default='manual-test.csv', help='test csv file')
parser.add_argument('--log-dir', type=str, default=package_root + 'logs/', help='logs root dir')
# parser.add_argument('--weights-dir', type=str, default=package_root + 'weights/', help='weights root dir')

# """ Train Parameters """
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--valid-frac', type=float, default=0.1, help='validation size fraction')
parser.add_argument('--test-frac', type=float, default=0.1, help='test size fraction')
# parser.add_argument('--amount-of-noise', type=float, default=0.12, help='probability of adding noise string')
# parser.add_argument('--name-length', type=int, default=60, help='fixed name length')

# """ Network Parameters"""
parser.add_argument('--hidden-size', type=int, default=256, help='hidden size')
parser.add_argument('--output-size', type=int, default=50, help='output size')
parser.add_argument('--num-layers', type=int, default=1,  help='number of rnn layers to use')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0, help='dropout ratio')
# parser.add_argument('--max-grad-norm', type=float, default=2,help='maximum gradient clipping normalization')
# parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5, help='probability of using teacher forcing')
parser.add_argument('--embedding-vec-size', type=int, default=100, help='char embedding vec size')
# parser.add_argument('--fixed-embeddings', action='store_true', default=False, help='embeddings are static')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 weight regularization')
parser.add_argument('--rnn-type', type=str, default="RNN", help='RNN cell type')
parser.add_argument('--bidirectional', action='store_true', default=False, help='use bi-directional rnn')
# parser.add_argument('--attention', action='store_true', default=False, help='use attention')



def _main():
    """ setup arguments """
    args = parser.parse_args()
    """ set random seed """
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if not args.cpu_only:
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.random_seed)
    """ run experiment """
    run_experiment(args)


if __name__ == "__main__":
    _main()

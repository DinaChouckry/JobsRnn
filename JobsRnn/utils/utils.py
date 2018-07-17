import subprocess

import torch
import shutil
import os
import pandas as pd
import numpy as np


def save_checkpoint(args, rnn_state, is_best):
    exp_weights_root_dir = args.weights_dir + args.exp_name + '/'
    os.makedirs(exp_weights_root_dir, exist_ok=True)
    rnn_filename = exp_weights_root_dir + 'checkpoint.pth.tar'
    torch.save(rnn_state, rnn_filename)
    if is_best:
        print('best beaten')
        shutil.copyfile(rnn_filename, exp_weights_root_dir + 'model_best.pth.tar')
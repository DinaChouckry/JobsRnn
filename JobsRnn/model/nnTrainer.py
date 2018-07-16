import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm
import random

class nnTrainer:
    def __init__(self, rnn, args, vocab, loader, logger, criterion , device):
        self.args = args
        self.rnn = rnn
        self.vocab = vocab
        self.loader = loader
        self.logger = logger
        self.criterion = criterion
        self.device = device
        return


    def __setup_batch(self, batch, data_time, start):
        batch_size = len(batch[0])
        job_id, seq_full_job, len_full_job = batch
        # data_time.update(time.time() - start)
        seq_full_job = seq_full_job.to(self.device)
        job_id = job_id.to(self.device)
        len_full_job = len_full_job.to(self.device)
        return batch_size, (seq_full_job, job_id, len_full_job)


    def __run_model(self, is_train , s_batch, rnn_optimizer, save_predictions):
        with torch.set_grad_enabled(is_train):
            seq_full_job, job_id, len_full_job = s_batch
            if is_train:
                rnn_optimizer.zero_grad()
            final_output, rnn_hidden = self.rnn(seq_full_job, len_full_job)
            topv, topi = final_output.topk(1)
            # rnn_output = rnn_output.transpose(0,1)
            job_id = job_id.type(torch.LongTensor)
            # print(topi.squeeze() , job_id)
            print("accuracy", np.sum((topi.squeeze() == job_id).numpy())/self.args.batch_size)
            loss = self.criterion(final_output, job_id)
            if is_train:
                loss.backward()
                rnn_optimizer.step()
        return loss.item()


    def __trainer(self, mode, data_loader, rnn_optimizer, epoch, logger, save_predictions=False, min_error=1e8):
        print("in -{0}- mode".format(mode))
        if mode == "train":
            is_train = True
            self.rnn.train()
        else:
            is_train = False
            self.rnn.eval()

        start = time.time()
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            batch_size, s_batch = self.__setup_batch(batch, start, start) #-->
            loss = self.__run_model(is_train, s_batch, rnn_optimizer, save_predictions)

        return loss


    def train_model(self, train_loader, rnn_optimizer, epoch, logger, val_min_error):
        return self.__trainer("train", train_loader, rnn_optimizer, epoch, logger,min_error=val_min_error)


    def validate_model(self, validate_loader, epoch, logger):
        return self.__trainer("validate", validate_loader, None, None, epoch, logger)


    def test_model(self, test_loader, epoch, logger, save_predictions=False):
        return self.__trainer("test", test_loader, None, None, epoch, logger, save_predictions=save_predictions)

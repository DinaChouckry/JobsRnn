import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DatasetInitializer:
    ## load input data from os folder all in one object
    def __init__(self, args):
        # kwargs = {'nrows': args.data_sample} if args.experimental else {}
        df = pd.read_csv(args.input_dir + args.train_csv)
        df = df.sample(frac=1, random_state=args.random_seed)
        test_frac = (len(df) * args.test_frac) / (len(df) - (len(df) * args.valid_frac))
        # generates from df an array
        self.train_x, self.valid_x = train_test_split(df.values, test_size=args.valid_frac,
                                                      random_state=args.random_seed)
        self.train_x, self.test_x = train_test_split(self.train_x, test_size=test_frac, random_state=args.random_seed)

        print("Data Loaded Successfully!")
        print("train_size: {0} \t valid_size: {1} \t test_size: {2}".format(len(self.train_x), len(self.valid_x),
                                                                            len(self.test_x)))


class DatasetSplittingGenerator(Dataset):
    """ Split data into train ,valid and test , assign it to Dataset  """

    def __init__(self, dataset_initialized, args, split_part_name, transform=None):
        """
        Args:
            args: global experiment arguments
            split_part_name: which data portion to be loaded
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.args = args
        self.split_part_name = split_part_name
        if self.split_part_name == 'train':
            self.x = dataset_initialized.train_x
        elif self.split_part_name == 'valid':
            self.x = dataset_initialized.valid_x
        elif self.split_part_name == 'test':
            self.x = dataset_initialized.test_x
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        job_id = self.x[idx, 0]
        full_job = self.x[idx, 1]
        pair = (full_job, job_id)

        if self.transform:
            pair = self.transform(pair)

        return pair

    def collate_fn(self, data):
        """
        Creates mini-batch tensors from (src_sent, tgt , src_seq ).
        We should build a custom collate_fn rather than using default collate_fn,
        because merging sequences (including padding) is not supported in default.
        Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

        Args:
            data: list of tuple (src_sents, tgt, src_seqs)
            - src_sents, tgt_sents: batch of original tokenized sentences
            - src_seqs: batch of original tokenized sentence ids
        Returns:
            - src_sents, tgt (tuple): batch of original tokenized sentences
            - src_seqs (variable): (max_src_len)
            - src_lens (tensor): (batch_size)

        """

        def _pad_sequences(seqs):
            lengths = [len(seq) for seq in seqs]
            padded_seqs = torch.zeros(len(seqs), max(lengths), dtype=torch.long)
            for i, seq in enumerate(seqs):
                end = lengths[i]
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
            return padded_seqs, torch.LongTensor(lengths)

        # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
        # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
        # *source* sequence, which is in the EncoderRNN

        data.sort(key=lambda x: len(x[0]), reverse=True)

        # Separate source and target sequences.
        str_full_job, job_id, seq_full_job = zip(*data)

        # Merge sequences (from tuple of 1D tensor to 2D tensor)
        seq_full_job, len_full_job = _pad_sequences(seq_full_job)

        # (batch, seq_len) => (seq_len, batch)
        seq_full_job = seq_full_job.transpose(0, 1)

        return str_full_job, job_id, seq_full_job, len_full_job

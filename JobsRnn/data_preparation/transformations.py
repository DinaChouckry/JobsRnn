import random as rn
from torch import nn
import numpy as np
from numpy.random.mtrand import rand, randint as random_randint, choice as random_choice
import pandas as pd
import nltk


# class dict_builder(object):
#     def __init__(self,args):
#         self.dict = []
#         self.df = DatasetInitializer(args).df
#
#     def build_dict(self):
#         self.df = self.df.dropna()
#         self.df['tokens']= self.df.apply(lambda row: nltk.word_tokenize(row['job']), axis=1)
#         self.dict = list(set([x for y in self.df['tokens'] for x in y]))
#
#     def get_dict(self):
#         if self.dict ==[]:
#             self.build_dict()
#         return self.dict



class transformation_helpers:

    @classmethod
    def build_dict(cls, df):
        df = df.dropna()
        df['tokens']= df.apply(lambda row: nltk.word_tokenize(row['job']), axis=1)
        words = list(set([x for y in df['tokens'] for x in y]))
        cls.vocab_list = ['<pad>', '<sos>', '<unk>', '<eos>'] + words
        cls.vocab = dict((c, i) for i, c in enumerate(cls.vocab_list))
        cls.inv_vocab = dict((i, c) for i, c in enumerate(cls.vocab_list))


    @classmethod
    def normalize(cls,df,series=None):
        df[series] = df[series].str.replace("[a-zA-Z0-9]","")
        df[series] = df[series].str.replace("ى","ي")
        df[series] = df[series].str.replace("ة","ه")
        df[series] = df[series].str.replace("چ","ج")
        for alf in ["آ","أ","إ"]:
            df[series] = df[series].str.replace(alf, "ا")
        return df

    @classmethod
    def string_to_int(cls, sentence, add_sos = True, add_eos=True):
        # switch sentence to indices and add <sos> , <eos>
        # sentence = list(sentence)
        # print('sentence 2 ',sentence)
        # TODO: switch back to error
        ints = list(map(lambda x: cls.vocab.get(x, cls.vocab['<unk>']), str(sentence).split()))
        ints = [cls.vocab['<sos>']] + ints if add_sos else ints
        ints = ints + [cls.vocab['<eos>']] if add_eos else ints
        return ints


    @classmethod
    def int_to_string(cls, ints):
        # switch indices back to sentence and remove  <sos> , <eos>
        sos_index = ints.index(cls.vocab['<sos>'])
        eos_index = ints.index(cls.vocab['<eos>']) if cls.vocab['<eos>'] in ints else -1
        sentence = [cls.inv_vocab.get(i, ".") for i in ints[sos_index+1:eos_index]]
        return sentence


    @classmethod
    def generate_embedding(cls, args):
        word_embedding = nn.Embedding(len(cls.vocab_list), args.embedding_vec_size, padding_idx=cls.vocab['<pad>'])
        # if args.fixed_embeddings:
        #     word_embedding.weight.requires_grad = False
        return word_embedding




class ToIdx:
    def __init__(self, args, add_sos=True, add_eos=True):
        self.args = args
        self.add_sos = add_sos
        self.add_eos = add_eos

    def __call__(self, pair):
        x_idx = transformation_helpers.string_to_int(pair[0], self.add_sos, self.add_eos)
        return (pair[0],pair[1],x_idx )










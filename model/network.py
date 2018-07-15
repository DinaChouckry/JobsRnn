import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
# from .model.model_utils import sequence_mask


class RNN(nn.Module):
    def __init__(self, embedding=None, rnn_type='RNN', hidden_size=128, num_layers=1, dropout=0.3, output_size=50 , bidirectional=True):
        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_directions
        self.embedding = embedding
        self.sentence_vec_size = self.embedding.embedding_dim
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(input_size=self.sentence_vec_size,
                                              hidden_size=self.hidden_size,
                                              num_layers=self.num_layers,
                                              dropout=self.dropout,
                                              bidirectional=self.bidirectional)
        self.output_size = output_size
        self.out = nn.Linear(hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, src_seqs, src_lens, hidden=None):
        """
        Args:
            - src_seqs: (max_src_len, batch_size)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """
        # (max_src_len, batch_size) => (max_src_len, batch_size, sentence_vec_size)
        emb = self.embedding(src_seqs)
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, src_lens)
        packed_outputs, hidden = self.rnn(packed_emb, hidden)
        rnn_output, output_lens = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        output = self.softmax(self.out(rnn_output))

        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        return output, hidden

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden
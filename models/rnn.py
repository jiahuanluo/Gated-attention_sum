import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
import math
import numpy as np


class rnn_encoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(rnn_encoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        self.hidden_size = config.hidden_size
        self.config = config

        if config.selfatt:
            if config.attention == 'None':
                self.attention = None
            elif config.attention == 'bahdanau':
                self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
            elif config.attention == 'luong':
                self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
            elif config.attention == 'luong_gate':
                self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size)
            elif config.attention == 'self_att':
                self.attention = models.MultiHeadAttention(n_head=8, d_model=config.hidden_size, d_k=64, d_v=64)

        if config.cell == 'gru':
            self.rnn1 = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size, dropout=config.dropout,
                               bidirectional=config.bidirectional, num_layers=1)
            self.rnn2 = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout,
                               bidirectional=config.bidirectional)
            self.rnn3 = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout,
                               bidirectional=config.bidirectional)
        else:
            self.rnn1 = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size, dropout=config.dropout,
                                bidirectional=config.bidirectional, num_layers=1)
            self.rnn2 = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout,
                                bidirectional=config.bidirectional)
            self.rnn3 = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout,
                                bidirectional=config.bidirectional)

    def forward(self, inputs, lengths):
        embs = pack(self.embedding(inputs), lengths)
        outputs1, state1 = self.rnn1(embs)
        outputs1 = unpack(outputs1)[0]  # (torch.Size([seq_length, batch_size, hidden_size * 2]))
        outputs1 = outputs1[:, :, :self.config.hidden_size] + outputs1[:, :, self.config.hidden_size:]
        outputs1, _ = self.attention(outputs1, outputs1, outputs1)
        outputs1 = outputs1.transpose(0, 1)
        outputs2, state2 = self.rnn2(outputs1, state1)
        outputs2 = outputs2[:, :, :self.config.hidden_size] + outputs2[:, :, self.config.hidden_size:]
        outputs2, _ = self.attention(outputs2, outputs2, outputs2)
        outputs2 = outputs2.transpose(0, 1)
        outputs3, state3 = self.rnn2(outputs2, state2)
        outputs3 = outputs3[:, :, :self.config.hidden_size] + outputs3[:, :, self.config.hidden_size:]
        outputs3, _ = self.attention(outputs3, outputs3, outputs3)
        outputs = outputs3.transpose(0, 1)
        if self.config.cell == 'gru':
            state = state2[:self.config.dec_num_layers]
        else:
            state = (torch.stack([state1[0][0], state2[0][0], state3[0][0]], 0),
                     torch.stack([state1[1][0], state2[1][0], state3[1][0]], 0))

        return outputs, state  # (seq_length, batch_size, hidden_size), ([num_layers, batch_size, hidden_size]， [num_layers, batch_size, hidden_size])


class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True):
        super(rnn_decoder, self).__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.emb_size)

        input_size = config.emb_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            if config.bi_dec:
                self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                                   num_layers=config.enc_num_layers, dropout=config.dropout,
                                   bidirectional=config.bidirectional)
            else:
                self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                       num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)
        self.linear_ = nn.Linear(config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size, prob=config.dropout)
        elif config.attention == 'self_att':
            self.attention = models.MultiHeadAttention(n_head=8, d_model=config.hidden_size, d_k=64, d_v=64)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, input, state):
        embs = self.embedding(input)
        output, state = self.rnn(embs,
                                 state)  # (batch_size, hidden_size), [(num_layer, batch_size, hidden_size), (num_layer, batch_size, hidden_size)]
        if self.config.bi_dec:
            output = output[-1:, :, :self.config.hidden_size] + output[-1:, :, self.config.hidden_size:]
            output = output.squeeze(0)
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                output, attn_weights = self.attention(output)
            elif self.config.attention == 'self_att':
                attn_input = output.unsqueeze(0)
                output, attn_weights = self.attention(attn_input, self.attention.context, self.attention.context)
                output = output.squeeze(1)
            else:
                output, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None

        output = self.compute_score(output)

        return output, state, attn_weights

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1

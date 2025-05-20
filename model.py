import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import wandb
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, rnn_type='LSTM',
                 dropout=0.2, bidirectional=False,use_attention=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.is_bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim // self.num_directions,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        if self.use_attention:
            return outputs,hidden
        else:
            return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, rnn_type='LSTM',
                 dropout=0.2,use_attention=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn_type = rnn_type
        self.use_attention = use_attention

        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embed_dim + hidden_dim if use_attention else embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.attn = None
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden_state, encoder_outputs=None,src_mask=None):
        embedded = self.embedding(input_token.unsqueeze(1))  # (B, 1, E)

        if self.use_attention and encoder_outputs is not None:
            # encoder_outputs: (B, T, H), hidden_state[0][-1]: (B, H)
            if self.rnn_type == 'LSTM':
                query = hidden_state[0][-1].unsqueeze(1)  # (B, 1, H)
            else:
                query = hidden_state[-1].unsqueeze(1)

            # Repeat query across time steps
            query = query.expand(-1, encoder_outputs.size(1), -1)  # (B, T, H)

            # Concatenate and compute attention weights
            energy = self.attn(torch.cat((encoder_outputs, query), dim=2))  # (B, T, 1)
            energy = energy.squeeze(2)
            if src_mask is not None:
                energy = energy.masked_fill(~src_mask, float('-inf'))
            attn_weights = F.softmax(energy, dim=1)  # (B, T)

            # Compute context vector
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B, 1, H)
            rnn_input = torch.cat((embedded, context), dim=2)  # (B, 1, E + H)
        else:
            rnn_input = embedded

        rnn_output, hidden = self.rnn(rnn_input, hidden_state)  # rnn_output: (B, 1, H)
        logits = self.fc_out(rnn_output.squeeze(1))  # (B, V)
        if self.use_attention:
            return logits, hidden, attn_weights  # Return attention weights
        else:
            return logits, hidden
        

class TransliterationModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim, hidden_dim,
                 enc_layers, dec_layers, rnn_type='LSTM', dropout=0.2, bidirectional=False,use_attention=False):
        super().__init__()
        self.encoder = Encoder(input_vocab_size, embed_dim, hidden_dim,
                                    enc_layers, rnn_type, dropout, bidirectional,use_attention)
        self.decoder = Decoder(output_vocab_size, embed_dim, hidden_dim,
                                     dec_layers, rnn_type, dropout,use_attention)
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.use_attention = use_attention

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        src_mask = (src != 0)
        if self.use_attention:
            enc_outputs, enc_hidden = self.encoder(src)
        else:
            enc_hidden = self.encoder(src)

        def merge_bidir_states(state):
            return torch.cat([state[::2], state[1::2]], dim=2)

        def pad_layers(state, target_layers):
            if state.size(0) == target_layers:
                return state
            pad = torch.zeros(target_layers - state.size(0), *state.shape[1:], device=state.device)
            return torch.cat([state, pad], dim=0)

        if self.rnn_type == 'LSTM':
            h, c = enc_hidden
            if self.bidirectional:
                h, c = merge_bidir_states(h), merge_bidir_states(c)
            h, c = pad_layers(h, self.dec_layers), pad_layers(c, self.dec_layers)
            dec_hidden = (h, c)
        else:
            h = enc_hidden
            if self.bidirectional:
                h = merge_bidir_states(h)
            h = pad_layers(h, self.dec_layers)
            dec_hidden = h

        dec_input = tgt[:, 0]  # Start token
        for t in range(1, tgt_len):
            if self.use_attention:
                output, dec_hidden, attn_weights = self.decoder(dec_input, dec_hidden, enc_outputs, src_mask)
                if t == 1:  # Only collect attention weights for visualization once
                    all_attn_weights = attn_weights.unsqueeze(1)  # (B, 1, src_len)
                else:
                    all_attn_weights = torch.cat((all_attn_weights, attn_weights.unsqueeze(1)), dim=1)
            else:
                output, dec_hidden= self.decoder(dec_input, dec_hidden)

            outputs[:, t] = output
            top1 = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = tgt[:, t] if teacher_force else top1

        if self.use_attention:
            return outputs, all_attn_weights  # Shape: (B, tgt_len-1, src_len)
        else:
            return outputs
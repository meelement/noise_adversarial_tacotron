import torch
from torch import nn as nn

from model.attention import LSA
from model.modules import PreNet


class Decoder(nn.Module):
    def __init__(self, n_mels, decoder_dims, lstm_dims, speaker_dims, noise_dims):
        super().__init__()
        self.max_r = 10
        self.r = None
        self.n_mels = n_mels
        self.prenet = PreNet(n_mels + noise_dims, fc2_dims=decoder_dims // 2)
        self.attn_net = LSA(decoder_dims)
        self.attn_rnn = nn.GRUCell(decoder_dims + decoder_dims // 2 + speaker_dims, decoder_dims)
        self.rnn_input = nn.Linear(2 * decoder_dims + speaker_dims + noise_dims, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)

    def zoneout(self, prev, current, p=0.1):
        device = prev.device
        assert prev.device == current.device
        mask = torch.zeros(prev.size(), device=device).bernoulli_(p)
        return prev * mask + current * (1.0 - mask)

    def forward(
            self,
            encoder_seq,
            encoder_seq_proj,
            prenet_in,
            hidden_states,
            cell_states,
            context_vec,
            t,
            phone_len,
            speaker_embedding,
            noise_embedding
    ):
        """
        :param encoder_seq: [Batch, Text, 2 x Encoder]
        :param encoder_seq_proj: [Batch, Text, Attention]
        :param prenet_in: [Batch, Mel]
        :param hidden_states: ([Batch, Decoder], 2x [Batch, LSTM])
        :param cell_states: ([Batch, LSTM], [Batch, LSTM])
        :param context_vec: [Batch, Context]
        :param t: int
        :param phone_len: the length of phone [Batch] LongTensor
        :param speaker_embedding: speaker embeddings [Batch, Speaker]
        :param noise_embedding: noise embeddings [Batch, Noise]
        :return: (mels, scores, hidden_states, cell_states, context_vec)
        """
        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # Concat the noise embedding inside PreNet
        prenet_in = torch.cat([prenet_in, noise_embedding], dim=-1) # [Batch, NMel + Noise]
        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in) # [Batch, PreOut]

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out, speaker_embedding], dim=-1) # [Batch, Context + PreOut + Speaker + Noise]
        attn_hidden = self.attn_rnn(attn_rnn_in, attn_hidden) # [Batch, Decoder]

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t, phone_len) # [Batch, 1, Text]

        # Dot product to create the context vector
        context_vec = scores @ encoder_seq # [Batch, 1, Context]
        context_vec = context_vec.squeeze(1) # [Batch, Context]

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden, speaker_embedding, noise_embedding], dim=1) # [Batch, Context + Decoder + Speaker + Noise]
        x = self.rnn_input(x) # [Batch, LSTM]
        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell)) # 2 x [Batch, LSTM]
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell)) # 2 x [Batch, LSTM]
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x) # [Batch, NMels x MaxR]
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r] # [Batch, NMels, R]
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden) # ([Batch, Decoder], [Batch, LSTM], [Batch, LSTM])
        cell_states = (rnn1_cell, rnn2_cell) # ([Batch, LSTM], [Batch, LSTM])

        return mels, scores, hidden_states, cell_states, context_vec

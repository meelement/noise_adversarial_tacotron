from torch import nn as nn
from model.modules import PreNet, CBHG
import torch
from hp import hook
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence, pack_sequence


def conv_len(l, kernel, stride, dilation):
    return (l - dilation * (kernel - 1) - 1 + stride) // stride


class CNNRNNEncoder(nn.Module):
    """
    Techniquely all the RNNs should mask away paddings.
    Practically this probably does not matter...
    """

    def __init__(self, in_channels, out_channels, encoder_dims):
        super(CNNRNNEncoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, encoder_dims * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_dims * 2, momentum=0.01, affine=False),
            nn.ReLU(),
            nn.Conv1d(encoder_dims * 2, encoder_dims * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_dims * 2, momentum=0.01, affine=False),
            nn.ReLU(),
            nn.Conv1d(encoder_dims * 2, encoder_dims, kernel_size=3, padding=1),
        )
        self.lstm = nn.LSTM(
            input_size=encoder_dims,
            hidden_size=encoder_dims,
            num_layers=2,
            bidirectional=True
        )
        self.mean_proj = nn.Linear(2 * encoder_dims, out_channels)
        self.mean_batch_norm = nn.BatchNorm1d(out_channels, momentum=0.01, affine=True)
        self.log_std_proj = nn.Linear(2 * encoder_dims, out_channels)

    def forward(self, x, l, conv_key="NULLKEY", encode_key="NULLKEY"):
        """
        :param x: Stuff to be encoded [Batch, InChannels, Time]
        :return: [Batch, OutChannels], [Batch, OutChannels] for mean and log_std
        """
        x = self.convs(x) # [Batch, D, Time]
        x = torch.transpose(x, 1, 2) # [Batch, Time, D]

        # Must apply packing to eliminate boundary effect in RNN:
        packed_x = pack_padded_sequence(x, l, batch_first=True)
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        hook[conv_key] = out[0]
        correction_ratio = torch.max(l) / l.float()
        x = torch.mean(out, dim=1) * correction_ratio.unsqueeze(-1) # [Batch, D x 2]
        ###############################################################
        latent_mean = self.mean_proj(x)
        latent_log_std = self.log_std_proj(x)
        latent_mean = self.mean_batch_norm(latent_mean)
        hook[encode_key] = latent_mean
        latent_log_std = torch.tanh(latent_log_std / 3.0) * 2.2 - 1.8
        return latent_mean, latent_log_std


class CNNEncoder(nn.Module):
    """
    Techniquely all the RNNs should mask away paddings.
    Practically this probably does not matter...
    """

    def __init__(self, in_channels, out_channels, encoder_dims):
        super(CNNEncoder, self).__init__()
        self.config = [
            (3, 1, 1), (3, 3, 2), (3, 2, 2), (3, 1, 1), (3, 1, 1)
        ]
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, encoder_dims, kernel_size=3, stride=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_dims, momentum=0.01, affine=True),
            nn.Conv1d(encoder_dims, encoder_dims, kernel_size=3, stride=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_dims, momentum=0.01, affine=True),
            nn.Conv1d(encoder_dims, encoder_dims, kernel_size=3, stride=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_dims, momentum=0.01, affine=True),
            nn.Conv1d(encoder_dims, encoder_dims, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_dims, momentum=0.01, affine=True),
            nn.Conv1d(encoder_dims, encoder_dims, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.mean_proj = nn.Linear(encoder_dims, out_channels)
        self.mean_batch_norm = nn.BatchNorm1d(out_channels, momentum=0.01, affine=True)
        self.log_std_proj = nn.Linear(encoder_dims, out_channels)

    def forward(self, x, l, conv_key="NULLKEY", encode_key="NULLKEY"):
        """
        :param x: Stuff to be encoded [Batch, InChannels, Time]
        :return: [Batch, OutChannels], [Batch, OutChannels] for mean and log_std
        """
        x = self.convs(x) # [Batch, EncoderChannel, Time]
        for kernel, stride, dilation in self.config:
            l = conv_len(l, kernel, stride, dilation)
        hook[conv_key] = x[0]
        mask = torch.arange(torch.max(l), device=x.device)[None, :] < l[:, None]
        correction_ratio = torch.max(l) / l.float() # [Batch]
        mask = mask.float() # [Batch, Time]

        x = x * mask.unsqueeze(dim=1)
        x = torch.mean(x, dim=-1) * correction_ratio.unsqueeze(-1)
        ###############################################################
        latent_mean = self.mean_proj(x)
        latent_log_std = self.log_std_proj(x)
        latent_mean = self.mean_batch_norm(latent_mean)
        latent_log_std = torch.tanh(latent_log_std / 3.0) * 4.0
        hook[encode_key] = latent_mean
        return latent_mean, latent_log_std


class Encoder(nn.Module):
    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims, dropout=dropout)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                         proj_channels=[cbhg_channels, cbhg_channels],
                         num_highways=num_highways)

    def forward(self, x):
        """
        :param x: [Batch, Text]
        :return: [Batch, Text, Encoder]
        """
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2) # [Batch, Embedding, Text]
        x = self.cbhg(x) # [Batch, Text, Encoder]
        return x

import torch
from torch import nn as nn
from torch.nn import functional as F


class PlainAttention(nn.Module):
    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):
        """
        Simplest stateless attention mechanism.
        :param encoder_seq_proj: [Batch, Text, Attention]
        :param query: [Batch, Attention]
        :param t: int
        :return: [Batch, 1, Text]
        """
        query_proj = self.W(query).unsqueeze(1) # [Batch, 1, Attention]
        u = self.v(torch.tanh(encoder_seq_proj + query_proj)) # [Batch, Text, 1]
        scores = F.softmax(u, dim=1) # [Batch, Text, 1]

        return scores.transpose(1, 2) # [Batch, 1, Text]


class LSA(nn.Module):
    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(2, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=False)
        self.L = nn.Linear(filters, attn_dim, bias=True)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None
        self.alpha = None
        self.attention_mask = None
        self.device = None

    def init_attention(self, encoder_seq_proj, phone_len):
        self.device = device = next(self.parameters()).device  # use same device as parameters
        Batch, Text, Attention = encoder_seq_proj.size()
        self.cumulative = torch.zeros(Batch, Text, device=device)
        self.attention = torch.zeros(Batch, Text, device=device)
        self.alpha = torch.zeros(Batch, Text, device=device)
        self.alpha[:, 0] = 1.0
        self.attention_mask = torch.arange(Text, device=self.device)[None, :] < phone_len[:, None]
        self.attention_mask = self.attention_mask.float()

    def forward(self, encoder_seq_proj, query, t, phone_len, enable_forward_bias=True):
        """
        Location sensitive attention
        :param encoder_seq_proj: [Batch, Text, Attention]
        :param query: [Batch, Attention]
        :param t: int
        :param phone_len: Long [Batch, Attention]
        :return: [Batch, 1, Text]
        """
        if t == 0: self.init_attention(encoder_seq_proj, phone_len)

        processed_query = self.W(query).unsqueeze(1) # [Batch, 1, Attention]

        location = torch.stack([self.cumulative, self.attention], dim=1) # [Batch, 2, Text]
        processed_loc = self.L(self.conv(location).transpose(1, 2)) # Convolution along Attention. [Batch, Text, Attention]

        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc)) # [Batch, Text, 1]
        u = u.squeeze(-1) # [Batch, Text]

        # Smooth Attention
        scores = torch.sigmoid(u) / torch.sigmoid(u).sum(dim=1, keepdim=True) # [Batch, Text]
        # scores = F.softmax(u, dim=1)
        self.attention = scores * self.attention_mask
        self.cumulative += self.attention
        # Introducing Forward Bias
        if enable_forward_bias:
            shift_alpha = F.pad(self.alpha[:, :-1], [1, 0, 0, 0], value=0.0)
            dual_alpha = F.pad(self.alpha[:, :-2], [2, 0, 0, 0], value=0.0)
            self.alpha = (shift_alpha + dual_alpha + self.alpha  + 1e-7) * self.attention
            self.alpha = self.alpha / torch.sum(self.alpha, dim=-1, keepdim=True)
            return self.alpha.unsqueeze(-1).transpose(1, 2) # [Batch, 1, Text]
        else:
            return self.attention.unsqueeze(-1).transpose(1, 2) # [Batch, 1, Text]
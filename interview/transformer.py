import torch

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, d_model=512, drop_rate=0.0):
        """

        Args:
            d_model:
            drop_rate:
        """
        super().__init__()
        self.d_model = d_model

        # # linear transform
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.qk_scale = d_model ** -0.5
        self.proj = nn.Linear(d_model, d_model)  # output transform
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, len_seq, _ = x.shape

        # 1.calculate q k v
        qkv = self.qkv(x).reshape(batch_size, len_seq, 3, self.d_model).permute(2, 0, 1, 3)  # (3, b, len_seq, d_model)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, len_seq, d_model)

        # 2.calculate attention score
        # Q * K^T / sqrt(d_k)
        attn_scores = q @ k.transpose(-2, -1)  # (batch, len_seq, len_seq)
        attn_scores *= self.qk_scale

        # 3.mask attention score
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 4. softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. calculate weight sum
        output = attn_weights @ v  # (batch, len_seq, d_model)

        # 6.output project
        output = self.proj(output)

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, drop_rate=0.0):
        """
        multi-head self-attention
        Args:
            d_model:
            num_heads:
            drop_rate:
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # linear transform
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.qk_scale = d_model ** -0.5
        self.proj = nn.Linear(d_model, d_model)  # output transform
        self.dropout = nn.Dropout(drop_rate)

    def combine_heads(self, x):
        """

        Args:
            x: [batch_size, num_heads, len_seq, d_k]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, len_seq, _ = x.shape
        x = x.transpose(1, 2).reshape(batch_size, len_seq, self.d_model)
        return x

    def forward(self, x, mask=None):
        """
        Args:
            x:  [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, len_seq, _ = x.shape

        # 1.calculate q k v and split head
        # (3, batch, num_head, len_seq, d_head)
        qkv = self.qkv(x).reshape(batch_size, len_seq, 3, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (batch, num_head, len_seq, d_head)

        # 2.calculate attention score
        # Q * K^T / sqrt(d_k)
        attn_scores = q @ k.transpose(-2, -1)  # (batch, num_head, len_seq, len_seq)
        attn_scores *= self.qk_scale

        # 3.mask attention score
        if mask is not None:
            if mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 4. softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. calculate weight sum
        output = torch.matmul(attn_weights, v)  # [batch, num_heads, len_seq, d_head]

        # 6. concat multi_head
        output = self.combine_heads(output)  # [batch, seq_len, d_model]

        # 7. 输出变换
        # 6.output project
        output = self.proj(output)

        return output


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    # self-attention
    self_attention = SelfAttention(d_model)
    output = self_attention(x)
    print(f'input shape: {x.shape}')
    print(f'output shape: {output.shape}')

    # multi-head self-attention
    multi_self_attention = MultiHeadSelfAttention(d_model, num_heads)
    output_multi = multi_self_attention(x)
    print(f'input shape: {x.shape}')
    print(f'output shape: {output.shape}')

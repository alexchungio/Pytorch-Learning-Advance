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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head=8):
        super().__init__()
        assert embed_dim % num_head == 0

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        # q k v projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # out projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, len_seq_q, _ = query.shape
        _, len_seq_k, _ = key.shape

        # projection
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # (b, num_head, len_seq_q, head_dim)
        Q = Q.view(batch_size, len_seq_q, self.num_head, self.head_dim).transpose(1, 2)
        # # (b, num_head, len_seq_k, head_dim)
        K = K.view(batch_size, len_seq_k, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, len_seq_k, self.num_head, self.head_dim).transpose(1, 2)

        # cal attention score
        # (b, num_head, len_seq_q, len_seq_k)
        attn_score = Q @ K.transpose(-2, -1)
        attn_score /= self.head_dim ** -0.5

        if attn_mask is not None:
            attn_score = attn_score.masked_fill(~attn_mask, float('-inf'))

        # normalize
        attn_score = torch.softmax(attn_score, -1)

        # dot value
        # (b, num_head, len_seq_q, head_dim)
        out = torch.matmul(attn_score, V)

        # combine
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        out = self.out_proj(out)

        return out, attn_score


if __name__ == "__main__":
    batch_size = 2
    len_seq_q = 10
    len_seq_k = 20
    d_model = 512
    num_heads = 8

    # x = torch.randn(batch_size, seq_len, d_model)

    # # self-attention
    # self_attention = SelfAttention(d_model)
    # output = self_attention(x)
    # print(f'input shape: {x.shape}')
    # print(f'output shape: {output.shape}')
    #
    # # multi-head self-attention
    # multi_self_attention = MultiHeadSelfAttention(d_model, num_heads)
    # output_multi = multi_self_attention(x)
    # print(f'input shape: {x.shape}')
    # print(f'output shape: {output.shape}')

    q = torch.randn(batch_size, len_seq_q, d_model)
    k = torch.randn(batch_size, len_seq_k, d_model)
    v = torch.randn(batch_size, len_seq_k, d_model)

    mha = MultiHeadAttention(
        embed_dim=d_model,
        num_head=num_heads,
    )

    output, weights = mha(q, k, v)
    print(output.shape)
    print(weights.shape)


import math
import torch
from torch import nn


# Positional encoding using trigonometric functions.
class PositionalEncoding(nn.Module):
    # Make sure max_len is large enough to cover all the num steps of input text.
    def __init__(self,
                 embedding_size,
                 max_len=1000,
                 dropout_ratio=0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        self.P = torch.zeros((1, max_len, embedding_size))
        # (max_len,) / (embedding_size/2) -> (max_len, embedding_size/2)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
                10000,
                torch.arange(0, embedding_size, 2, dtype=torch.float32) /
                embedding_size)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # (batch_size, num_steps, embedding_size)
        return self.dropout(X + self.P[:, :X.shape[1], :].to(X.device))


# Dot attention
class DotProductAttention(nn.Module):

    def __init__(self, dropout=0.5) -> None:
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # query: (batch_size, q_num, hidden_num)
        # key: (batch_size, k_num, hidden_num)
        # value: (batch_size, v_num, hidden_num)
        d = queries.shape[-1]
        # score: (batch_size, q_num, k_num) && k_num == v_num
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # attention_weight: (batch_size, q_num, k_num)
        self.attention_weight = self.masked_attention(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weight), values)

    def masked_attention(self, scores, valid_lens):
        # scores: (batch_size, q_size, k_size)
        # valid_lens: (batch_size, 1)
        if valid_lens == None:
            return nn.functional.softmax(scores, dim=-1)
        if valid_lens.dim == 1:
            valid_lens = torch.repeat_interleave(valid_lens, scores.shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # mask each sequence.
        default_value = -1e6
        for idx in range(scores.shape[0]):
            valid_len = valid_lens[idx]
            scores[idx][valid_len:] = default_value
        return nn.functional.softmax(scores, dim=-1)


# Multi Head Attention.
class MultiHeadAttention(nn.Module):

    def __init__(self,
                 q_size,
                 k_size,
                 v_size,
                 hidden_num,
                 head_num,
                 dropout=0.5) -> None:
        super(MultiHeadAttention, self).__init__()
        self.Wq = nn.Linear(q_size, hidden_num)
        self.Wk = nn.Linear(k_size, hidden_num)
        self.Wv = nn.Linear(v_size, hidden_num)
        self.Wo = nn.Linear(hidden_num, hidden_num)
        self.head_num = head_num
        self.attention = DotProductAttention(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries: (batch_size, q_num, q_size) -> (batch, head_num, q_size, hidden_num/head_num)
        # keys: (batch_size, k_num, k_size) -> (batch, head_num, k_size, hidden_num/head_num)
        # values: (batch_size, v_num, v_size) -> (batch, head_num, v_size, hidden_num/head_num)
        queries = self.transfer_qkv(self.Wq(queries), self.head_num)
        keys = self.transfer_qkv(self.Wk(keys), self.head_num)
        values = self.transfer_qkv(self.Wv(values), self.head_num)
        # expand valid_len dimension to fit batch_size*head_num
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 self.head_num,
                                                 dim=0)
        # output: (batch_size*head_num, q_num, hidden_num/head_num) -> (batch_size, q_num, hidden_num)
        output = self.attention(queries, keys, values, valid_lens)
        output = self.Wo(self.transfer_output(output, self.head_num))
        return output

    def transfer_qkv(self, X, head_num):
        X = X.reshape(X.shape[0], X.shape[1], head_num, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(
            -1, X.shape[2], X.shape[3]
        )  # (batch_size * head_num, q/k/v_num, hidden_num/head_num)

    def transfer_output(self, X, head_num):
        X = X.reshape(-1, head_num, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1],
                         -1)  # (batch_size, q_num, hidden_num)


# Add Norm
class AddNorm(nn.Module):

    def __init__(self, dropout_rate) -> None:
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # Normalization for each token in text sentence.
        self.layer_norm = nn.LayerNorm()

    def forward(self, X, Y):
        return self.layer_norm(X + self.dropout(Y))


# Positionwise ffn.
class PositionwiseFFN(nn.Module):

    def __init__(self, ffn_input_num, ffn_hidden_num, ffn_output_num) -> None:
        super(PositionwiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_input_num, ffn_hidden_num)
        self.dense2 = nn.Linear(ffn_hidden_num, ffn_output_num)

    def forward(self, X):
        return self.dense2(self.dense1(X))


if __name__ == '__main__':
    print('test positional encoding...')
    positional_encoding = PositionalEncoding(10)
    test = torch.ones((2, 5, 10))
    test = positional_encoding(test)
    print(test)
    print(test.shape)

    print('test dot product attention...')
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10,
                                                           4).repeat(2, 1, 1)
    dot_product_attention = DotProductAttention(dropout=0.5)
    test = dot_product_attention(queries, keys, values, None)
    print(test)
    print(test.shape)

    print('test multi head attention...')
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(
        X, Y, Y, valid_lens).shape)  # (batch_size, num_queries, hidden_nums)

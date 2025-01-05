import math
import model_utils
import torch
from torch import nn


# Transformer encoder layer
class EncoderLayer(nn.module):

    def __init__(self,
                 q_size,
                 k_size,
                 v_size,
                 hidden_num,
                 ffn_input_num,
                 ffn_hidden_num,
                 ffn_output_num,
                 head_num=5,
                 drop_out=0.5) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = model_utils.MultiHeadAttention(q_size,
                                                        k_size,
                                                        v_size,
                                                        hidden_num,
                                                        head_num,
                                                        dropout=0.5)
        self.add_norm_1 = model_utils.AddNorm(drop_out)
        self.positionwise_ffn = model_utils.PositionwiseFFN(
            ffn_input_num, ffn_hidden_num, ffn_output_num)
        self.add_norm_2 = model_utils.AddNorm(drop_out)

    def forward(self, src_input, src_valid_lens):
        # src_input.shape: (batch_size, num_steps, src_vocab_size)
        # res.shape: (batch_size, num_steps, hidden_num)
        Y = self.attention(src_input, src_input, src_input, src_valid_lens)
        X = self.add_norm_1(src_input, Y)
        Y = self.positionwise_ffn(X)
        res = self.add_norm_2(X, Y)
        return res


# Transformer encoder
class Encoder(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 q_size,
                 k_size,
                 v_size,
                 hidden_num,
                 ffn_input_num,
                 ffn_hidden_num,
                 ffn_output_num,
                 head_num,
                 layer_num=6,
                 dropout=0.5) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, hidden_num)
        self.positional_encoding = model_utils.PositionalEncoding(dropout)
        self.hidden_num = hidden_num
        self.enc_layers = nn.Sequential()
        self.attention_weight = [None for i in range(layer_num)]
        for idx in range(layer_num):
            self.enc_layers.add_module(
                "block" + str(idx),
                EncoderLayer(q_size, k_size, v_size, hidden_num, ffn_input_num,
                             ffn_hidden_num, ffn_output_num, head_num,
                             dropout))

    def forward(self, src_input, src_valid_lens):
        X = self.embedding(src_input)
        X = self.positional_encoding(X * math.sqrt(self.hidden_num))
        for idx, layer in self.enc_layers:
            X = layer(X, src_valid_lens)
            self.attention_weight[
                idx] = layer.attention.attention.attention_weight
        return X


# Transformer decoder layer
class DecoderLayer(nn.module):

    def __init__(self,
                 q_size,
                 k_size,
                 v_size,
                 hidden_num,
                 ffn_input_num,
                 ffn_hidden_num,
                 ffn_output_num,
                 head_num,
                 idx,
                 dropout=0.5) -> None:
        super(DecoderLayer, self).__init__()
        self.idx = idx
        self.attention_1 = model_utils.MultiHeadAttention(
            q_size, k_size, v_size, hidden_num, head_num, dropout)
        self.add_norm_1 = model_utils.AddNorm(dropout)
        self.attention_2 = model_utils.MultiHeadAttention(
            q_size, k_size, v_size, hidden_num, head_num, dropout)
        self.add_norm_2 = model_utils.AddNorm(dropout)
        self.positionwise_ffn = model_utils.PositionwiseFFN(
            ffn_input_num, ffn_hidden_num, ffn_output_num)
        self.add_norm_3 = model_utils.AddNorm(dropout)

    def forward(self, tgt_input, enc_output, src_valid_lens, tgt_valid_lens):
        # output: (batch_size, num_steps, hidden_num)
        Y = self.attention_1(tgt_input, tgt_input, tgt_input, tgt_valid_lens)
        X = self.add_norm_1(tgt_input, Y)
        Y = self.attention_2(X, enc_output, enc_output, src_valid_lens)
        X = self.add_norm_2(X, Y)
        output = self.add_norm_3(X, self.positionwise_ffn(X))
        return output


# Transformer decoder
class Decoder(nn.Module):

    def __init__(self,
                 tgt_vocab_size,
                 q_size,
                 k_size,
                 v_size,
                 hidden_num,
                 ffn_input_num,
                 ffn_hidden_num,
                 ffn_output_num,
                 head_num,
                 layer_num=6,
                 dropout=0.5) -> None:
        super(Decoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.embedding = nn.Embedding(tgt_vocab_size, self.hidden_num)
        self.positional_encoding = model_utils.PositionalEncoding(dropout)
        self.dec_layers = nn.Sequential()
        for idx in range(self.layer_num):
            self.dec_layers.add_module(
                "block" + str(idx),
                DecoderLayer(q_size, k_size, v_size, hidden_num, ffn_input_num,
                             ffn_hidden_num, ffn_output_num, head_num,
                             dropout))
        self.ffn = nn.Linear(ffn_output_num, tgt_vocab_size)

    def forward(self, tgt_input, enc_output, src_valid_lens, tgt_valid_lens):
        X = self.embedding(tgt_input)
        X = self.positional_encoding(X * math.sqrt(self.hidden_num))
        # init because there are two attention layers in each decoder blocker.
        self.attention_weight = [[None, None] for i in range(self.layer_num)]
        for idx, layer in enumerate(self.dec_layers):
            X = layer(X, enc_output, src_valid_lens, tgt_valid_lens)
            self.attention_weight[idx][
                0] = layer.attention_1.attention.attention_weight
            self.attention_weight[idx][
                1] = layer.attention_2.attention.attention_weight
        output = self.ffn(X)
        return output


# Transformer model
class Transformer(nn.Module):

    def __init__(self, hidden_num, src_vocab_size, tgt_vocan_size) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Linear(hidden_num, tgt_vocan_size)

    def forward(self, src, target, src_valid_lens, target_lens):
        enc_output = self.encoder(src)  # (batch_size, num_steps, num_hiddens)
        output = self.decoder(target, enc_output)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    # basic unit test
    pass

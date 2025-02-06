import math
import model_utils
import torch
from torch import nn


# Transformer encoder layer
class EncoderLayer(nn.Module):

    def __init__(self,
                 q_size,
                 k_size,
                 v_size,
                 hidden_num,
                 norm_shape,
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
        self.add_norm_1 = model_utils.AddNorm(norm_shape, drop_out)
        self.positionwise_ffn = model_utils.PositionwiseFFN(
            ffn_input_num, ffn_hidden_num, ffn_output_num)
        self.add_norm_2 = model_utils.AddNorm(norm_shape, drop_out)

    def forward(self, src_input, src_valid_lens):
        # src_input.shape: (batch_size, num_steps, hidden_num)
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
                 norm_shape,
                 ffn_input_num,
                 ffn_hidden_num,
                 ffn_output_num,
                 head_num,
                 layer_num=6,
                 dropout=0.5) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, hidden_num)
        self.positional_encoding = model_utils.PositionalEncoding(hidden_num)
        self.hidden_num = hidden_num
        self.enc_layers = nn.Sequential()
        self.attention_weight = [None for i in range(layer_num)]
        for idx in range(layer_num):
            self.enc_layers.add_module(
                "block" + str(idx),
                EncoderLayer(q_size, k_size, v_size, hidden_num, norm_shape,
                             ffn_input_num, ffn_hidden_num, ffn_output_num,
                             head_num, dropout))

    def forward(self, src_input, src_valid_lens):
        X = self.embedding(src_input)
        X = self.positional_encoding(X * math.sqrt(self.hidden_num))
        for idx, layer in enumerate(self.enc_layers):
            X = layer(X, src_valid_lens)
            self.attention_weight[
                idx] = layer.attention.attention.attention_weight
        return X


# Transformer decoder layer
class DecoderLayer(nn.Module):

    def __init__(self,
                 q_size,
                 k_size,
                 v_size,
                 hidden_num,
                 norm_shape,
                 ffn_input_num,
                 ffn_hidden_num,
                 ffn_output_num,
                 head_num,
                 dropout=0.5) -> None:
        super(DecoderLayer, self).__init__()
        self.attention_1 = model_utils.MultiHeadAttention(
            q_size, k_size, v_size, hidden_num, head_num, dropout)
        self.add_norm_1 = model_utils.AddNorm(norm_shape, dropout)
        self.attention_2 = model_utils.MultiHeadAttention(
            q_size, k_size, v_size, hidden_num, head_num, dropout)
        self.add_norm_2 = model_utils.AddNorm(norm_shape, dropout)
        self.positionwise_ffn = model_utils.PositionwiseFFN(
            ffn_input_num, ffn_hidden_num, ffn_output_num)
        self.add_norm_3 = model_utils.AddNorm(norm_shape, dropout)

    def forward(self, tgt_input, enc_output, src_valid_lens):
        # init tgt valid lens (batch_size, num_steps)
        tgt_valid_lens = None
        if self.training:
            tgt_valid_lens = torch.arange(1,
                                          tgt_input.shape[1] + 1,
                                          device=tgt_input.device).repeat(
                                              tgt_input.shape[0], 1)
        Y = self.attention_1(tgt_input, tgt_input, tgt_input, tgt_valid_lens)
        X = self.add_norm_1(tgt_input, Y)
        Y = self.attention_2(X, enc_output, enc_output, src_valid_lens)
        X = self.add_norm_2(X, Y)
        #  (batch_size, num_steps, hidden_num)
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
                 norm_shape,
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
        self.positional_encoding = model_utils.PositionalEncoding(hidden_num)
        self.dec_layers = nn.Sequential()
        for idx in range(self.layer_num):
            self.dec_layers.add_module(
                "block" + str(idx),
                DecoderLayer(q_size, k_size, v_size, hidden_num, norm_shape,
                             ffn_input_num, ffn_hidden_num, ffn_output_num,
                             head_num, dropout))
        self.ffn = nn.Linear(ffn_output_num, tgt_vocab_size)

    def forward(self, tgt_input, enc_output, src_valid_lens):
        X = self.embedding(tgt_input)
        X = self.positional_encoding(X * math.sqrt(self.hidden_num))
        # init because there are two attention layers in each decoder blocker.
        self.attention_weight = [[None, None] for i in range(self.layer_num)]
        for idx, layer in enumerate(self.dec_layers):
            X = layer(X, enc_output, src_valid_lens)
            self.attention_weight[idx][
                0] = layer.attention_1.attention.attention_weight
            self.attention_weight[idx][
                1] = layer.attention_2.attention.attention_weight
        output = self.ffn(X)
        return output


# Transformer model
class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 q_size,
                 k_size,
                 v_size,
                 hidden_num,
                 norm_shape,
                 ffn_input_num,
                 ffn_hidden_num,
                 ffn_output_num,
                 head_num,
                 enc_layer_num=6,
                 dec_layer_num=6,
                 dropout=0.5) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, q_size, k_size, v_size,
                               hidden_num, norm_shape, ffn_input_num,
                               ffn_hidden_num, ffn_output_num, head_num,
                               enc_layer_num, dropout)
        self.decoder = Decoder(tgt_vocab_size, q_size, k_size, v_size,
                               hidden_num, norm_shape, ffn_input_num,
                               ffn_hidden_num, ffn_output_num, head_num,
                               dec_layer_num, dropout)

    def forward(self, src_input, tgt_input, src_valid_lens):
        enc_output = self.encoder(
            src_input, src_valid_lens)  # (batch_size, num_steps, num_hiddens)
        output = self.decoder(
            tgt_input, enc_output,
            src_valid_lens)  # (batch_size, num_steps, num_hiddens)
        return output


if __name__ == '__main__':
    src_vocab_size = 200
    tgt_vocab_size = 400
    hidden_num = 50
    head_num = 5
    layer_num = 6
    dropout = 0.5
    # test encoder layer
    print('\ntesting encoder layer:')
    src_input = torch.ones((5, 30, 50))
    src_valid_lens = torch.tensor([5, 6, 7, 8, 9])
    enc_layer = EncoderLayer(hidden_num, hidden_num, hidden_num, hidden_num,
                             hidden_num, hidden_num, hidden_num, hidden_num)
    enc_layer_output = enc_layer(src_input, src_valid_lens)
    print('src_input:', src_input.shape)
    print('enc layer output: ',
          enc_layer_output.shape)  # (batch_size, num_steps, hidden_num)

    # test encoder
    print('\ntesting encoder:')
    src_input = torch.ones((5, 30), dtype=torch.int)
    valid_len = torch.tensor([5, 6, 7, 8, 9])
    encoder = Encoder(src_vocab_size, hidden_num, hidden_num, hidden_num,
                      hidden_num, hidden_num, hidden_num, hidden_num,
                      hidden_num, head_num, layer_num, dropout)
    enc_output = encoder(src_input, src_valid_lens)
    print('src_input:', src_input.shape)
    print('enc output:', enc_output.shape)

    # test decoder layer
    print('\ntesting decoder layer:')
    tgt_input = torch.ones((5, 30, 50), dtype=torch.float32)
    valid_len = torch.tensor([5, 6, 7, 8, 9])
    decoder_layer = DecoderLayer(hidden_num, hidden_num, hidden_num,
                                 hidden_num, hidden_num, hidden_num,
                                 hidden_num, hidden_num, head_num, dropout)
    dec_layer_output = decoder_layer(tgt_input, enc_output, src_valid_lens)
    print('dec_input:', tgt_input.shape)
    print('dec layer output:', dec_layer_output.shape)

    # test decoder
    print('\ntesting decoder:')
    tgt_input = torch.ones((5, 40), dtype=torch.int)
    valid_len = torch.tensor([5, 6, 7, 8, 9])
    decoder = Decoder(tgt_vocab_size, hidden_num, hidden_num, hidden_num,
                      hidden_num, hidden_num, hidden_num, hidden_num,
                      hidden_num, head_num, layer_num, dropout)
    enc_output = decoder(tgt_input, enc_output, src_valid_lens)
    print('src_input:', tgt_input.shape)
    print('enc output:', enc_output.shape)

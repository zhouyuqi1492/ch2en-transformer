import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils as utils
import math
import pickle
from model import Transformer
import model_utils
from preprocess import OneHotTokenizer, TranslationDataset
from tqdm import tqdm

# parameters
batch_size = 2
shuffle = True
epochs = 50
cuda = torch.cuda.is_available()
seed = 2025
max_len = 128
hidden_num = 128
lr = 0.5
optimizer = 'adam'


# init params
def xavier_init_weights(model):
    if type(model) == nn.Linear:
        nn.init.xavier_uniform_(model.weight)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):

    def __init__(self):
        super(MaskedSoftmaxCELoss, self).__init__()

    def forward(self, pred, label, valid_len):
        # pred: (batch_size,num_steps,vocab_size)
        # label: (batch_size,num_steps)
        # valid_len的形状：(batch_size,)
        weights = torch.ones_like(label, dtype=torch.float32)
        max_len = pred.shape[1]
        mask = torch.arange((max_len),
                            dtype=torch.float32,
                            device=weights.device)[None, :] < valid_len[:,
                                                                        None]
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_func():
    # init dataloader
    print('init dataset')
    src_data_file_path = 'data/src_train.txt'
    tgt_data_file_path = 'data/tgt_train.txt'
    src_tokenizer_file_path = 'data/src_tokenizer.pkl'
    with open(src_tokenizer_file_path, 'rb') as rbfile:
        src_tokenizer = pickle.load(rbfile)
    tgt_tokenizer_file_path = 'data/tgt_tokenizer.pkl'
    with open(tgt_tokenizer_file_path, 'rb') as rbfile:
        tgt_tokenizer = pickle.load(rbfile)
    dataset = TranslationDataset(src_data_file_path, src_tokenizer,
                                 tgt_data_file_path, tgt_tokenizer, max_len)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)
    # model
    print('init model')
    src_vocab_size = len(src_tokenizer)
    tgt_vocab_size = len(tgt_tokenizer)
    model = Transformer(src_vocab_size,
                        tgt_vocab_size,
                        q_size=hidden_num,
                        k_size=hidden_num,
                        v_size=hidden_num,
                        hidden_num=hidden_num,
                        norm_shape=hidden_num,
                        ffn_input_num=hidden_num,
                        ffn_hidden_num=hidden_num,
                        ffn_output_num=hidden_num,
                        head_num=4,
                        enc_layer_num=6,
                        dec_layer_num=6,
                        dropout=0.5)
    model.apply(xavier_init_weights)
    # optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    model.train()
    # traning process
    print('start training: ')
    for epoch in range(epochs):
        loss_total = 0
        print('epoch: ', epoch)
        for batch in tqdm(dataloader,
                          desc=f'Epoch {epoch + 1}/{epochs}',
                          unit='batch'):
            src_batch = batch['src']
            tgt_batch = batch['tgt']
            src_valid_len_batch = batch['src_valid_len']
            tgt_valid_len_batch = batch['tgt_valid_len']
            # (batch_size, num_steps, tgt_vocab_size)
            pred = model(src_batch, tgt_batch, src_valid_len_batch)
            l = loss(pred, tgt_batch, tgt_valid_len_batch)
            l.sum().backward()
            optimizer.zero_grad()
            clip_value = 1.0  # 设置梯度的最大值
            utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            loss_total += l.sum()
        if epoch + 1 % 10:
            print('epoch: ', epoch, loss_total)


def test_func():
    raise NotImplementedError


def evaluator():
    raise NotImplementedError


if __name__ == '__main__':
    train_func()

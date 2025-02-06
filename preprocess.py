import jieba
import math
import pickle
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

data_path = 'data/news-commentary-v13-zh-en.txt'
# reserved tokens
pad_token = '<pad>'
ukn_token = '<ukn>'
bos_token = '<bos>'
eos_token = '<eos>'
reserved_tokens = [pad_token, ukn_token, bos_token, eos_token]


class OneHotTokenizer:
    # Collecting vocabs for each language.
    def __init__(self, counter, word2idx, idx2word, is_for_ch) -> None:
        self.counter = counter
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.is_for_ch = is_for_ch

    def __len__(self):
        return len(self.word2idx)

    def convert_word_to_idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return self.word2idx[ukn_token]

    def convert_idx_to_word(self, index):
        return self.idx2word[index]

    def encode(self, sentence, max_len):
        sentence = sentence.replace('\n', '')
        vocab_size = len(list(self.counter))
        if self.is_for_ch:
            words = list(jieba.cut(sentence))
        else:
            words = sentence.split(' ')
        words = [bos_token] + words + [eos_token]
        indices = [self.convert_word_to_idx(word) for word in words]
        valid_len = min(len(indices), max_len)
        if len(indices) > max_len:
            indices = indices[:max_len -
                              1] + [self.convert_word_to_idx(eos_token)]
        elif len(indices) < max_len:
            indices += [self.convert_word_to_idx(pad_token)
                        ] * (max_len - len(indices))
        # one_hot_tensor = torch.zeros(max_len, vocab_size, dtype=torch.long)
        # one_hot_tensor[range(max_len), indices] = 1
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        valid_len_tensor = torch.tensor(valid_len, dtype=torch.long)
        return indices_tensor, valid_len_tensor

    def decode(self, indices):
        words = []
        for index in indices:
            word = self.idx2word[index]
            if word == eos_token:
                break
            words.append(self.convert_idx_to_word(index))
        return ' '.join(words)


class TranslationDataset(Dataset):

    def __init__(self, src_data_file_path, src_tokenizer, tgt_data_file_path,
                 tgt_tokenizer, max_len) -> None:
        super(TranslationDataset, self).__init__()
        # init sentences and tokenizer
        self.src_sentences = None
        with open(src_data_file_path, 'r') as rfile:
            self.src_sentences = rfile.readlines()
        self.tgt_sentences = None
        with open(tgt_data_file_path, 'r') as rfile:
            self.tgt_sentences = rfile.readlines()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        # get sentences
        src_sentence = self.src_sentences[index].replace('\n', '')
        tgt_sentence = self.tgt_sentences[index].replace('\n', '')
        # tokenizer processing
        src_tensor, src_valid_len_tensor = self.src_tokenizer.encode(
            src_sentence, self.max_len)
        tgt_tensor, tgt_valid_len_tensor = self.tgt_tokenizer.encode(
            tgt_sentence, self.max_len)
        return {
            'src': src_tensor,
            'tgt': tgt_tensor,
            'src_valid_len': src_valid_len_tensor,
            'tgt_valid_len': tgt_valid_len_tensor
        }


# Get ch-en pair corpus.
def GetParallelCorpus():
    print('reading examples:')
    source_sentences = []
    target_sentences = []
    with open(data_path, 'r') as date_file:
        line_count = 0
        for idx, line in enumerate(date_file):
            line = line.replace('\n', '')
            source_sentence, target_sentence = line.split('\t')
            if idx % 10000 == 0:
                print(idx, source_sentence, target_sentence)
            if (len(line) == 0):
                continue
            line_count += 1
            source_sentences.append(source_sentence)
            target_sentences.append(target_sentence.lower())
    print("source corpus len: ", len(source_sentences))
    print("target corpus len: ", len(target_sentences))
    return source_sentences, target_sentences


# TODO(Yukizh): Build tokenizer in one function by using is_for_ch param.
def BuildCHTokenizer(ch_dataset, min_freq, reserved_tokens):
    print('Building Chinese tokenizer...')
    counter = Counter()
    word2idx, idx2word = {}, {}
    # Add reserved tokens.
    for token in reserved_tokens:
        temp_idx = len(word2idx)
        word2idx[token] = temp_idx
        idx2word[temp_idx] = token
    # Add from src copurs.
    for idx in tqdm(range(len(ch_dataset))):
        line = ch_dataset[idx].replace('\n', '')
        seg_list = jieba.cut(line, cut_all=True)
        counter.update(list(seg_list))
    for token in counter:
        if counter[token] < min_freq:
            continue
        temp_idx = len(word2idx)
        word2idx[token] = temp_idx
        idx2word[temp_idx] = token
    print('Vocab size: ', len(word2idx))
    print('Building Chinese tokenizer success!')
    return OneHotTokenizer(counter, word2idx, idx2word, is_for_ch=True)


def BuildENTokenizer(en_dataset, min_freq, reserved_tokens):
    print('Building English tokenizer ...')
    counter = Counter()
    word2idx = {}
    idx2word = {}
    # Add reserved tokens.
    for token in reserved_tokens:
        temp_idx = len(word2idx)
        word2idx[token] = temp_idx
        idx2word[temp_idx] = token
    # Add from tgt corpus.
    for idx in tqdm(range(len(en_dataset))):
        text = en_dataset[idx]
        seg_list = text.split(' ')
        counter.update(seg_list)
    for token in counter:
        if counter[token] < min_freq:
            continue
        temp_idx = len(word2idx)
        word2idx[token] = temp_idx
        idx2word[temp_idx] = token
    print('Vocab size: ', len(word2idx))
    print('Building English tokenizer success!')
    return OneHotTokenizer(counter, word2idx, idx2word, is_for_ch=False)


def GetDatesetInfo():
    src_corpus, tgt_corpus = GetParallelCorpus()
    print('-----')
    print('Source data: ')
    src_max_len = -math.inf
    src_min_len = math.inf
    src_average_len = 0
    count = 0
    for item in src_corpus:
        seg_list = list(jieba.cut(item))
        if len(seg_list) > src_max_len:
            src_max_len = len(seg_list)
        if len(seg_list) < src_min_len:
            src_min_len = len(seg_list)
            print(item)
            print(seg_list)
        if count == 0:
            src_average_len = len(seg_list)
            count += 1
            continue
        src_average_len = src_average_len + (len(seg_list) -
                                             src_average_len) / count
    print('max_len={}, min_len={}, average_len={}'.format(
        src_max_len, src_min_len, src_average_len))
    print('-----')
    print('Target data: ')
    tgt_max_len = -math.inf
    tgt_min_len = math.inf
    tgt_average_len = 0
    for item in tgt_corpus:
        seg_list = item.split(' ')
        if len(seg_list) > tgt_max_len:
            tgt_max_len = len(seg_list)
        if len(seg_list) < tgt_min_len:
            tgt_min_len = len(seg_list)
            print(seg_list)
        if count == 0:
            tgt_average_len = len(seg_list)
            count += 1
            continue
        tgt_average_len = tgt_average_len + (len(seg_list) -
                                             tgt_average_len) / count
    print('max_len={}, min_len={}, average_len={}'.format(
        tgt_max_len, tgt_min_len, tgt_average_len))


if __name__ == '__main__':
    # Get corpus.
    src_corpus, tgt_corpus = GetParallelCorpus()

    # Build vocabs.
    src_tokenizer = BuildCHTokenizer(src_corpus,
                                     min_freq=1,
                                     reserved_tokens=reserved_tokens)
    tgt_tokenizer = BuildENTokenizer(tgt_corpus,
                                     min_freq=1,
                                     reserved_tokens=reserved_tokens)
    with open('data/src_tokenizer.pkl', 'wb') as wfile:
        pickle.dump(src_tokenizer, wfile)
    with open('data/tgt_tokenizer.pkl', 'wb') as wfile:
        pickle.dump(tgt_tokenizer, wfile)

    # Split dataset.
    src_train, src_test, tgt_train, tgt_test = train_test_split(src_corpus,
                                                                tgt_corpus,
                                                                test_size=0.1)
    if len(src_train) != len(tgt_train) or len(src_test) != len(tgt_test):
        print('[Error] src_size != tgt_size ')
    print("train dataset size: ", len(src_train))
    print("test dataset size: ", len(src_test))
    with open('data/src_train.txt', 'w') as wfile:
        for line in src_train:
            wfile.write(line + '\n')
    with open('data/tgt_train.txt', 'w') as wfile:
        for line in tgt_train:
            wfile.write(line + '\n')
    with open('data/src_test.txt', 'w') as wfile:
        for line in src_test:
            wfile.write(line + '\n')
    with open('data/tgt_test.txt', 'w') as wfile:
        for line in tgt_test:
            wfile.write(line + '\n')

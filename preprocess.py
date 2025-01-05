import jieba
import math
import pickle
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

data_path = 'data/news-commentary-v13-zh-en.txt'
# reserved tokens
pad_token = '<pad>'
ukn_token = '<ukn>'
bos_token = '<bos>'
eos_token = '<eos>'
reserved_tokens = [pad_token, ukn_token, bos_token, eos_token]


class Vocab:
    # Collecting vocabs for each language.
    def __init__(self, counter, word2idx, idx2word) -> None:
        self.counter = counter
        self.word2idx = word2idx
        self.idx2word = idx2word

    def __len__(self):
        return len(self.word2idx)

    def ConvertTextToIdx(self, sentence, word2idx):
        return [
            word2idx[word] if word in word2idx else word2idx[ukn_token]
            for word in sentence
        ]

    def ConvertIdx2Text(self, sentence):
        words = []
        for i in sentence:
            word = self.idx2word[i]
            if word == eos_token:
                break
            words.append(self.idx2word[i])
        return ' '.join(words)


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


# TODO(Yukizh): Consider split numbers into single char.
def BuildCHVocab(ch_dataset, min_freq, reserved_tokens):
    print('Building Chinese vocabs...')
    counter = Counter()
    word2idx, idx2word = {}, {}
    # Add reserved tokens.
    for token in reserved_tokens:
        temp_idx = len(word2idx)
        word2idx[token] = temp_idx
        idx2word[temp_idx] = token
    # Add from src copurs.
    for idx in tqdm(range(len(ch_dataset))):
        seg_list = jieba.cut(ch_dataset[idx], cut_all=True)
        counter.update(list(seg_list))
    for token in counter:
        if counter[token] < min_freq:
            continue
        temp_idx = len(word2idx)
        word2idx[token] = temp_idx
        idx2word[temp_idx] = token
    print('Vocab size: ', len(word2idx))
    print('Building Chinese vocabs success!')
    return Vocab(counter, word2idx, idx2word)


def BuildENVocab(en_dataset, min_freq, reserved_tokens):
    print('Building English vocabs ...')
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
    print('Building English vocabs success!')
    return Vocab(counter, word2idx, idx2word)


def CalculateDatesetStatistics():
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
    src_vocab = BuildCHVocab(src_corpus,
                             min_freq=1,
                             reserved_tokens=reserved_tokens)
    tgt_vocab = BuildENVocab(tgt_corpus,
                             min_freq=1,
                             reserved_tokens=reserved_tokens)
    with open('data/src_vocab.pkl', 'wb') as wfile:
        pickle.dump(src_vocab, wfile)
    with open('data/tgt_vocab.pkl', 'wb') as wfile:
        pickle.dump(tgt_vocab, wfile)

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

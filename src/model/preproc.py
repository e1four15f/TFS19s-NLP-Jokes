import sqlite3
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import string


ALPHABET = 'abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890 ' + string.punctuation


def read_data(db_path, duplicates=True):
    conn = sqlite3.connect(db_path)
    curr = conn.cursor()

    raw_texts = []

    for row in tqdm(curr.execute("SELECT * FROM jokes")):
        raw_texts.append(row[1])

    if duplicates:
        length = len(raw_texts)
        raw_texts = raw_texts[:length//2]

    return raw_texts


class Vocab:
    def __init__(self, data):
        self.alphabet = ALPHABET
        self.char2idx, self.idx2char = self.build_vocab()

    def build_vocab(self):
        self.char2idx = {char: idx + 4 for idx, char in enumerate(self.alphabet)}
        self.char2idx['<START>'] = 0
        self.char2idx['<END>'] = 1
        self.char2idx['<PAD>'] = 2
        self.char2idx['<UNK>'] = 3
        self.idx2char = {value: key for key, value in self.char2idx.items()}
        return self.char2idx, self.idx2char

    def data2vocab(self, data):
        self.cv = CountVectorizer(analyzer='char')
        self.cv.fit(data)
        return self.cv

    def preprocess(self, data):
        """
        By now it do nothing, however if you want you can do it
        """
        return data

    def start_end_pad(self, data, sentence_size=None):
        data_ = []
        for twit in tqdm(data):
            # replace all whitespace characters by space only
            twit_ = twit.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace)))
            twit_ = self.tokenize(twit_)
            twit_ = [self.char2idx['<START>'], ] + twit_ + [self.char2idx['<END>'], ]
            if isinstance(sentence_size, int):
                twit_len = len(twit_)
                if twit_len < sentence_size:
                    twit_ += [self.char2idx['<PAD>'], ] * (sentence_size - twit_len)
                elif twit_len >= sentence_size:
                    twit_ = twit_[:sentence_size]
                    twit_[-1] = self.char2idx['<END>']
            data_.append(twit_)
        return data_

    def tokenize(self, sequence):
        return [self.char2idx[char]
                if char in self.alphabet else self.char2idx['<UNK>']
                for char in sequence.lower()]

    def detokenize(self, sequence):
        return ''.join([self.idx2char[idx] for idx in sequence])

    def __len__(self):
        return len(self.char2idx)

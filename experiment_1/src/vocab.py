import re


class Vocab:
    def __init__(self, data):
        self.alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 —!\n",.:;-'
        self.char2idx = {char: idx + 4 for idx, char in enumerate(self.alphabet)}
        for idx, token in enumerate(['<START>', '<END>', '<PAD>', '<UNK>']):
            self.char2idx[token] = idx
        self.idx2char = {value: key for key, value in self.char2idx.items()}

    def tokenize(self, sequence):
        sequence = sequence.lower()
        sequence = re.sub('(#[' + self.alphabet + ']+)+', '', sequence)
        sequence = re.findall('[' + self.alphabet + ']', sequence)
        return [self.char2idx[char] for char in sequence]
    
    def detokenize(self, sequence):
        return ''.join([self.idx2char[idx] for idx in sequence])
    
    def __len__(self):
        return len(self.char2idx)
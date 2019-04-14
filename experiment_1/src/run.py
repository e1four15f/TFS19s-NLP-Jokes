import tensorflow as tf
import pandas as pd
import numpy as np


from src.vocab import *
from src.model import *
from src.feed import *

class Run:
    def __init__(self, data_path='jokes.csv', model_dir='lmm11111'):
        self.build_vocab(data_path)
        self.load_model(model_dir)
    
    def build_vocab(self, data_path='jokes.csv'):
        self.data_path = data_path   
        self.vocab = Vocab(pd.read_csv(data_path)['text'])
        self.min_len, self.max_len = 40, 200
            
    def load_model(self, model_dir='lmm11111'):
        self.model_dir = model_dir
        self.params = {
            'vocab_size': len(self.vocab),
            'vocab': self.vocab,
            'train_size': 0,
            'num_layers': 2,
            'embedding_dim': 128,
            'lstm_hidden_dim': 256,
            'dropout_keep_prob': 0.7,
            'max_iter': self.max_len,
            'initial_state_is_zero': False,
            'beam_width': 5,
            'batch_size': 128,
            'num_epochs': 5000,
            'learning_rate': 1e-3,
            'clip': 0.5
        }

        config = tf.estimator.RunConfig(save_checkpoints_steps=1000,
                                        model_dir=self.model_dir,
                                        save_summary_steps=5)

        self.model = tf.estimator.Estimator(model_fn=model_fn, params=self.params, config=config)


    def generate_jokes(self, count=5):
        random_sequences = []
        for i in range(count):
            sequence_len = np.random.randint(self.min_len, self.max_len)
            sequence = np.random.choice(list(self.vocab.alphabet), size=sequence_len)
            random_sequences.append(''.join(sequence))

        data = [self.vocab.tokenize(sequence) for sequence in random_sequences]
        output = self.model.predict(lambda: input_fn(data, params=self.params, vocab=self.vocab, mode='predict'))

        for i in output:
            joke = self.vocab.detokenize(i['ids'][i['ids'] != 1])
            print('\n' + joke + '\n')
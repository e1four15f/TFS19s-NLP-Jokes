from .model import Model

import pandas as pd

from fastai.text import TextList, TokenizeProcessor, Tokenizer, NumericalizeProcessor, language_model_learner, AWD_LSTM
import torch
torch.cuda.set_device(0)


class ULMFiT_Model(Model):

    def __init__(self, path):
        texts = pd.read_csv(path + '/jokes_extended_vk_anekdot_preproc.csv', index_col=0)
        texts.dropna(inplace=True)
        data = TextList.from_df(texts, 
                        processor=[TokenizeProcessor(tokenizer=Tokenizer(lang="xx")), 
                                     NumericalizeProcessor(min_freq=2, max_vocab=60000)])\
                                    .split_by_rand_pct(.1)\
                                    .label_for_lm()\
                                    .databunch(bs=64)

        self.learn = language_model_learner(data=data, arch=AWD_LSTM, pretrained=None)
        self.learn.load_pretrained(path + '/ulmfit/bestmodel_tune.pth', path + '/ulmfit/bestmodel_tune_itos.pkl')

    def generate(self, temperature=1.0, text=''):
        raw_data = self.learn.predict(text=text, n_words=200, temperature=temperature)
        raw_data = raw_data if text else raw_data[1:]
    
        jokes, joke = [], ''
        for word in raw_data.split(' '):
            if word == 'xxbos':
                jokes.append(joke)
                if text:
                  break
                joke = ''
            else:
                joke += word + ' '
        
        result = ''
        for i, joke in enumerate(jokes):
            result += f'Шутка {i+1}\n{joke}\n\n'
        return result
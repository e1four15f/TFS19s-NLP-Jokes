from .model import Model

from textgenrnn import textgenrnn
from keras.backend import clear_session


class TextgenRNN_Model(Model):

    def __init__(self, path):
        clear_session()
        self._textgen = textgenrnn(weights_path=path + '/textgenrnn/colaboratory_weights.hdf5',
                                   vocab_path=path + '/textgenrnn/colaboratory_vocab.json',
                                   config_path=path + '/textgenrnn/colaboratory_config.json')
        
    def generate(self, temperature=1.0):
        jokes = self._textgen.generate(3, return_as_list=True,
                               temperature=1.0, max_gen_length=150)
        result = ''
        for i, joke in enumerate(jokes):
            result += f'Шутка {i+1}\n{joke}\n\n'
        return result
import tensorflow as tf
import functools

from models.src.vocab import Vocab


def create_data_dict(data, mode):
    data_dict = {
            'sequences': data[0], 
            'lengths': data[1]
        }

    if mode == 'spellcheck':
        data_dict['spellcheck'] = True

    return data_dict


def generator_fn(data, vocab):
    for sequence in data:
        sequence = [vocab.char2idx['<START>']] + sequence + [vocab.char2idx['<END>']]
        yield (sequence, len(sequence))


def input_fn(data, params, vocab, mode):
    shapes = ([None], ())
    types = (tf.int32, tf.int32)
    defaults = (vocab.char2idx['<PAD>'], 0)

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data, vocab), 
                                             output_shapes=shapes, output_types=types)
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=params['train_size'], reshuffle_each_iteration=True)
        dataset = dataset.repeat(params['num_epochs'])

    dataset = dataset.padded_batch(params['batch_size'], shapes, defaults)
    dataset = dataset.map(lambda *x: create_data_dict(x, mode))
    dataset = dataset.prefetch(buffer_size=1) 
    return dataset
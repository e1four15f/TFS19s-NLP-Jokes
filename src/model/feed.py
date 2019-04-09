from . import preproc
import tensorflow as tf
import functools
from numpy import random


"""
source will be in encoder, however target in decoder
source and target are used when they are different 
"""

class Feed:

    def __init__(self):
        pass

    def parse_fn(self, line_before, line_after, params):
        """
        Parse given lines
        :param line_before: string to encode, list of ints
        :param line_after: string to decode, list of ints
        :param params: dict, must contain vocab
        :return:
        """
        source = line_before
        learn_target = line_after + [params['encoder_vocab']['<END>'], ] \
                       + [params['encoder_vocab']['<PAD>'], ]
        target = [params['encoder_vocab']['<START>'], ] + line_after \
                 + [params['encoder_vocab']['<END>'], ]

        return (source, len(source)), (target, learn_target, len(line_after)+2)

    def generator_fn(self, docs_before, docs_after, params):
        for doc_before, doc_after in zip(docs_before, docs_after):
            yield self.parse_fn(doc_before, doc_after, params)

    def train_input_fn_(self, docs_before, docs_after, params):
        shapes = (([None], ()), ([None], [None], ()))
        types = ((tf.int32, tf.int32), (tf.int32, tf.int32, tf.int32))

        # TODO: encoder and decoder vocab should differ
        padding_values = ((params['encoder_vocab']['<PAD>'], 0),
                          (params['encoder_vocab']['<PAD>'], params['encoder_vocab']['<PAD>'], 0))

        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.generator_fn, docs_before, docs_after, params),
            output_types=types,
            output_shapes=shapes
        )
        dataset = dataset.repeat(params.get('epochs', 50))
        dataset = dataset.padded_batch(params.get('batch_size', 128), shapes, padding_values)
        return dataset.prefetch(1)

    def input_fn(self, docs_before, docs_after, params, mode='train'):
        if mode=='train':
            return functools.partial(self.train_input_fn_, docs_before, docs_after, params)

        raise NotImplementedError()

    def random_gen(self, params, n_chars=10, n_docs=10):
        vocab_id_min = 4
        vocab_id_max = params['encoder_vocab_size']
        for _ in range(n_docs):
            doc = random.randint(vocab_id_min, vocab_id_max, n_chars)
            yield (doc, n_chars), (doc+[5, 6], doc+[5, 6], n_chars)

    def random_input_fn_(self, params, n=10, n_docs=10):
        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.random_gen, params, n, n_docs),
            output_shapes= (([None], ()), ([None], [None], ())),
            output_types= ((tf.int32, tf.int32), (tf.int32, tf.int32, tf.int32))
        )
        dataset = dataset.batch(n_docs).repeat(1)
        return dataset.prefetch(1)

    def predict_input_fn(self, params, n=10):
        return functools.partial(self.random_input_fn_, params, n)
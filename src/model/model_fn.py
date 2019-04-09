import tensorflow as tf


# TODO: all functions in tf.nn.rnn_cell are deprecated, replace them by their analogies from Keras


def encoder(embeddings, length, params):
    lstm_size = params['encoder_lstm_size']
    num_layers = params['encoder_num_layers']
    encoder_final_context_dims = params['encoder_final_context_dims']
    encoder_vocab_size = params['encoder_vocab_size']
    state_size = params['encoder_state_size'] if 'encoder_state_size' in params \
        else params['encoder_lstm_size']
    batch_size = tf.shape(embeddings)[0]

    if params.get('initial_state_zero', False):
        tf.logging.info('Initialize cell states as zeros')

        # Don't quite understand what state size will be if i don't define initial state and other params of LSTMCell
        # cell_fw_initial_state = [
        #     tf.zeros((batch_size, state_size)) for _ in range(num_layers)
        # ]
        # cell_bw_initial_state = [
        #     tf.zeros((batch_size, state_size)) for _ in range(num_layers)
        # ]
        cell_bw_initial_state = None
        cell_fw_initial_state = None
    else:
        '''Haven't tried to use it, may contain mistakes'''

        cell_fw_initial_state = [
            tf.get_variable(f'trainable_cell_fw_initial_state_{i}',
                            shape=[batch_size, state_size],
                            initializer=tf.random.uniform([batch_size, state_size])
                            )
            for i in range(num_layers)
        ]
        cell_bw_initial_state = [
            tf.get_variable(f'trainable_cell_bw_initial_state_{i}',
                            shape=[batch_size, state_size],
                            initializer=tf.random.uniform([batch_size, state_size])
                            )
            for i in range(num_layers)
        ]

    multilayer_encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
        [tf.contrib.rnn.GRUBlockCellV2(lstm_size) for _ in range(num_layers)])
    multilayer_encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
        [tf.contrib.rnn.GRUBlockCellV2(lstm_size) for _ in range(num_layers)])

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=multilayer_encoder_cell_fw,
        cell_bw=multilayer_encoder_cell_bw,
        inputs=embeddings,
        sequence_length=length,
        initial_state_fw=cell_fw_initial_state,
        initial_state_bw=cell_bw_initial_state,
        dtype=tf.float32
    )

    output = tf.concat(outputs, axis=-1)
    encoder_output = tf.layers.dense(output, encoder_vocab_size)  # by now it doesn't use

    # prepare outputs of each cell in each layer
    state_fw, state_bw = states
    cells = []
    for fw, bw in zip(state_fw, state_bw):
        state = tf.concat([fw, bw], axis=-1)  # resulted shape will be [None, fw.shape[0]+bw.shape[0]]
        cells += [tf.layers.dense(state, encoder_final_context_dims)]  # instead of encoder_vocab_size maybe it is better to use another
    encoder_state = tuple(cells)
    return encoder_output, encoder_state


def get_decoder_cell(params):
    lstm_size = params['decoder_lstm_size']
    num_layers = params['encoder_num_layers']


    # decoder_base_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
    multi_layer_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.contrib.rnn.GRUBlockCellV2(lstm_size) for _ in range(num_layers)])

    return multi_layer_decoder_cell


def model_fn(features, labels, mode, params):
    """
    In this case features are tokens that need to be converted
    and labels are tokens which has to be decoded
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """
    encoder_vocab = params['encoder_vocab']
    assert len(encoder_vocab) == params['encoder_vocab_size']
    decoder_vocab = params['decoder_vocab']
    assert len(decoder_vocab) == params['decoder_vocab_size']

    source, source_length = features
    batch_size = tf.shape(source)[0]

    encoder_embeddings = tf.get_variable('encoder_embeddings',
                                         shape=[params['encoder_vocab_size'], params['encoder_emb_size']],
                                         # initializer=tf.random_uniform(
                                         #     [params['encoder_vocab_size'], params['encoder_emb_size']])
                                         initializer=tf.random_uniform_initializer(-1.0, 1.0)
                                         )
    source_embeddings = tf.nn.embedding_lookup(encoder_embeddings, source)

    # encode
    encoder_output, encoder_state = encoder(source_embeddings, source_length, params)

    # decoder
    decoder_cell = get_decoder_cell(params)
    decoder_initial_state = encoder_state

    if not params.get("use_encoder_embeddings", False):
        decoder_embeddings = encoder_embeddings
    else:
        decoder_embeddings = tf.get_variable("decoder_embeddings",
                                             shape=[params['decoder_vocab_size'], params['decoder_emb_size']],
                                             # initializer=tf.random_uniform(
                                             #     [params['decoder_vocab_size'], params['decoder_emb_size']])
                                             initializer=tf.random_uniform_initializer(-1.0, 1.0)
                                             )

    decoder_projection_layer = tf.layers.Dense(params['decoder_vocab_size'], use_bias=False)

    if mode == tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL:
        target, learn_target, target_length = labels
        target_embeddings = tf.nn.embedding_lookup(decoder_embeddings, target)
        learn_target_embeddings = tf.nn.embedding_lookup(decoder_embeddings, learn_target)
        train_helper = tf.contrib.seq2seq.TrainingHelper(target_embeddings, target_length)

        train_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=train_helper,
            initial_state=decoder_initial_state,
            output_layer=decoder_projection_layer
        )

        train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)
        train_output = train_outputs.rnn_output
        train_sample_id = train_outputs.sample_id

        '''
        return tensor of shape [batch_size, max(length for length in target_length)]
        with true in mask[i, length[i]] for i in range(batch_size)
        '''
        masks = tf.sequence_mask(lengths=target_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=train_output,
            targets=learn_target,
            weights=masks
        )

        metrics = {
            'acc': tf.metrics.accuracy(target, train_sample_id, masks)
        }
        for metric_name, value in metrics.items():
            tf.summary.scalar(metric_name, value[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm = tf.clip_by_global_norm(grads, params.get('clip', .5))
            train_op = optimizer.apply_gradients(zip(grads, vs), global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.PREDICT:

        # to generate random jokes, decoder_initial_state shape same as encoder_state shape
        decoder_initial_state = encoder_output

        # prediction_initial_state = tf.contrib.seq2seq.tile_batch(decoder_initial_state, 4)
        prediction_initial_state = decoder_initial_state
        tf.logging.info(decoder_embeddings)
        tf.logging.info(tf.fill([batch_size], tf.to_int32(decoder_vocab['<START>'])))
        tf.logging.info(tf.to_int32(decoder_vocab['<END>']))
        tf.logging.info(prediction_initial_state)
        prediction_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=decoder_embeddings,
            start_tokens=tf.fill([batch_size], tf.to_int32(decoder_vocab['<START>'])),
            end_token=tf.to_int32(decoder_vocab['<END>']),
            initial_state=prediction_initial_state,
            beam_width=4,
            output_layer=decoder_projection_layer
        )
        prediction_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            prediction_decoder,
            maximum_iterations=params['decoder_max_len']
        )

        reverse_decoder_vocab_tensor = tf.constant(
            [params['reverse_decoder_vocab'][i] for i in range(params['decoder_vocab_size'])]
        )
        reverse_decoder_lookup = tf.contrib.lookup.index_to_string_from_tensor(
            reverse_decoder_vocab_tensor,
            default_value="<SOME MISTAKE>"
        )
        # predicted_strings = reverse_decoder_lookup(tf.to_int64(prediction_output.sample_id))
        predicted_strings = reverse_decoder_lookup(tf.to_int64(prediction_output[:,:,0]))
        predictions = {
            'ids': prediction_output.sample_id,
            'text': predicted_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
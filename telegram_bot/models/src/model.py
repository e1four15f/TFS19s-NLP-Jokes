import tensorflow as tf


def model_fn(features, mode, params):
    sequences, lengths = features['sequences'], features['lengths']
    vocab = params['vocab']
    # матрица эмбеддингов
    embeddings = tf.Variable(tf.random_uniform([params['vocab_size'], 
                                                params['embedding_dim']]))
    # --- encoder ---
    encoder_embeddings = tf.nn.embedding_lookup(embeddings, sequences[:, 1:])
    # (многослойная) LSTM
    # ...
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(params['lstm_hidden_dim']) 
                                                for _ in range(params['num_layers'])])
    
    # Опционально: обучаемое начальное состояние
    # (если не делаете, то убедитесь, что вы всегда инициализируете его нулями перед каждым прогоном)
    # ...
    batch_size = tf.shape(sequences)[0]
    if params['initial_state_is_zero']:
        encoder_initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    else:
        encoder_initial_state = []
        for _ in range(params['num_layers']):
            with tf.variable_scope('encoder_initial_state', reuse=tf.AUTO_REUSE):
                state_c = tf.tile(tf.get_variable('init_c', [1, params['lstm_hidden_dim']]), (batch_size, 1))
                state_h = tf.tile(tf.get_variable('init_h', [1, params['lstm_hidden_dim']]), (batch_size, 1))
            layer_state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)
            encoder_initial_state.append(layer_state)    
        encoder_initial_state = tuple(encoder_initial_state)
        
        
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embeddings, 
                                                       sequence_length=lengths-1,#!
                                                       initial_state=encoder_initial_state,
                                                       dtype=tf.float32)
    
    # --- decoder ---
    decoder_initial_state = encoder_state
    projection_layer = tf.layers.Dense(params['vocab_size'], use_bias=False)
        
    # режимы, в которых необходимо считать loss для готовых последовательностей (train, eval и spellcheck).
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL or \
            (mode == tf.estimator.ModeKeys.PREDICT and 'spellcheck' in features):
        # --- train ---
        lengths -= 1
        target_embeddings = tf.nn.embedding_lookup(embeddings, sequences)
        
        train_helper = tf.contrib.seq2seq.TrainingHelper(target_embeddings, lengths)
        
        train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, 
                                                        helper=train_helper, 
                                                        initial_state=decoder_initial_state, 
                                                        output_layer=projection_layer)
        
        train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder,
                                                                maximum_iterations=params['max_iter'], 
                                                                impute_finished=True)
        logits = train_outputs.rnn_output
        train_sample_id = train_outputs.sample_id
        
        # --- loss ---
        masks = tf.sequence_mask(lengths=lengths, dtype=tf.float32)
        
        # в режиме spellcheck нужно вернуть loss для каждого элемента последовательностей
        if mode == tf.estimator.ModeKeys.PREDICT:
            losses = tf.contrib.seq2seq.sequence_loss(logits=logits, 
                                                      targets=sequences[:, 1:], 
                                                      weights=masks, 
                                                      average_across_timesteps=False, 
                                                      average_across_batch=False)
            predictions = {
                'sequences': sequences[:, 1:],
                'lengths': lengths,
                'losses': losses
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                    targets=sequences[:, 1:], 
                                                    weights=masks,
                                                    average_across_timesteps=True, 
                                                    average_across_batch=True)
            
            # в режиме eval возвращаем усреднённый лосс
            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = {
                    'acc': tf.metrics.accuracy(sequences[:, 1:], train_sample_id, masks)
                }
                for metric_name, value in metrics.items():
                    tf.summary.scalar(metric_name, value[1])
                
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            
            # в режиме train ещё и обновляем обучаемые параметры
            elif mode == tf.estimator.ModeKeys.TRAIN:             
                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, _ = tf.clip_by_global_norm(grads, params['learning_rate'])
                train_op = optimizer.apply_gradients(zip(grads, vs), global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        
    # в режиме predict генерируем продолжения заданных последовательностей.
    elif mode == tf.estimator.ModeKeys.PREDICT and 'spellcheck' not in features:
        start_tokens = tf.fill([batch_size], vocab.char2idx['<START>']) 
        
        predictor_initial_state = tf.contrib.seq2seq.tile_batch(decoder_initial_state, 
                                                                multiplier=params['beam_width'])
        
        
        prediction_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cell,
            embedding=embeddings,
            start_tokens=start_tokens,
            end_token=vocab.char2idx['<END>'],
            initial_state=predictor_initial_state,
            beam_width=params['beam_width'],
            output_layer=projection_layer)
   
        prediction_output, _, _ = tf.contrib.seq2seq.dynamic_decode(prediction_decoder, 
                                                                    maximum_iterations=params['max_iter'])
        predicted_ids = prediction_output.predicted_ids[:,:,0]

        predictions = {
            'ids': predicted_ids,
        }
        
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
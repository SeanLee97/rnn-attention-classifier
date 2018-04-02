# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

import tensorflow as tf

class RNNAttention(object):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self._build_graph()

    def _build_graph(self):
        if self.mode != 'train':
            self.keep_prob = 1.0   
        else:
            self.keep_prob = self.args.dropout

        self._set_placeholders()
        self._build_rnn_cell()
        self._embedding_lookup()
        self._rnn()

        if self.args.use_attention:
            self._attention()

        self._compute_loss()

        if self.mode == 'train':
            self._build_train_op()     

    def _set_placeholders(self):
        with tf.variable_scope('embedding'):
            self.word_embedding = tf.get_variable("word_embedding", shape=[self.args.vocab_size, self.args.embed_size], dtype=tf.float32)
        
        with tf.variable_scope('placeholders'):
            self.word_input = tf.placeholder(name="word_input", shape=[None, None], dtype=tf.int32)
            self.seq_length = tf.placeholder(name="seq_len", shape=[None], dtype=tf.int32)
            self.label = tf.placeholder(name="labels", shape=[None], dtype=tf.int32)

    def _build_rnn_cell(self):
        if self.args.rnn_type == 'lstm':
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([self._get_lstm_cell() for _ in range(self.args.num_layers)])
        else:
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([self._get_gru_cell() for _ in range(self.args.num_layers)])

    def _get_gru_cell(self):
        with tf.variable_scope('gru_cell'):
            rnn_cell = tf.contrib.rnn.GRUCell(num_units=self.args.hidden_size)
            if self.keep_prob < 1:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            return rnn_cell

    def _get_lstm_cell(self):
        with tf.variable_scope('lstm_cell'):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.args.hidden_size, forget_bias=1.0)
            if self.keep_prob < 1:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            return rnn_cell

    def _embedding_lookup(self):
        with tf.variable_scope('lookup'):
            self.word_emb_feature = tf.nn.embedding_lookup(self.word_embedding,
                                                           self.word_input,
                                                           name='word_emb_feature',
                                                           validate_indices=True)
    def _rnn(self):
        with tf.variable_scope('rnn_block'):
            if not self.args.bi_rnn:
                # 
                self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(cell=self.rnn_cell,
                                                                     inputs=self.word_emb_feature,
                                                                     sequence_length=self.seq_length,
                                                                     dtype=tf.float32)
                if self.args.rnn_type == 'lstm':
                    self.rnn_state = self.rnn_state[self.args.num_layers-1][1]
            else:
                # bidirectional
                ((fw_outputs, bw_outputs), (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.rnn_cell,
                                                                                                   cell_bw=self.rnn_cell,
                                                                                                   inputs=self.word_emb_feature,
                                                                                                   sequence_length=self.seq_length,
                                                                                                   dtype=tf.float32)
                self.rnn_outputs = tf.concat(values=(fw_outputs, bw_outputs), axis=2, name='concat_output')

                if self.args.rnn_type == 'lstm':
                    self.rnn_state = tf.concat(values=(fw_state[self.args.num_layers-1][1], bw_state[self.args.num_layers-1][1]), axis=1, name='concat_state')
                    # self.rnn_state = fw_state[1] + bw_state[1]
                elif self.args.rnn_type == 'gru':
                    self.rnn_state = tf.concat(values=(fw_state[self.args.num_layers-1], bw_state[self.args.num_layers-1]), axis=1, name='concat_state')
                    # self.rnn_state = fw_state + bw_state

    def _attention(self):
        with tf.variable_scope('attention'):
            attention_vector = tf.get_variable(name='attention_vector',
                                               shape=[self.args.attn_size],
                                               dtype=tf.float32)

            layer_proj = tf.layers.dense(inputs=self.rnn_outputs,
                                                   units=self.args.attn_size,
                                                   activation=tf.nn.tanh,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   name='attn_layer')

            attended_vector = tf.tensordot(layer_proj, attention_vector, axes=[[2], [0]])
            attention_weights = tf.expand_dims(tf.nn.softmax(attended_vector), -1)

            weighted_input = tf.matmul(self.rnn_outputs, attention_weights, transpose_a=True)
            self.attention_output = tf.squeeze(weighted_input, axis=2)

    def _compute_loss(self):

        with tf.variable_scope('dense_layers'):
            with tf.variable_scope('dropout'):
                if self.args.use_attention:
                    sentence_vector = tf.nn.dropout(self.attention_output, keep_prob=self.keep_prob, name='attention_vector_dropout')
                else:
                    sentence_vector = tf.nn.dropout(self.rnn_state, keep_prob=self.keep_prob, name='rnn_state_dropout')

            output1 = tf.layers.dense(inputs=sentence_vector,
                                      units=self.args.label_size,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                      bias_initializer=tf.constant_initializer(0.01),
                                      name='layer_1')

        with tf.variable_scope('last_layer'):
            self.logits = tf.layers.dense(inputs=output1,
                                     units=self.args.label_size,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01),
                                     name='layer_logit')

            with tf.name_scope('pred_acc'):
                with tf.name_scope('prediction'):
                    self.proba = tf.nn.softmax(self.logits, name='softmax_probability')
                    self.prediction = tf.cast(tf.argmax(input=self.proba, axis=1, name='prediction'), dtype=tf.int32)
                    correct_prediction = tf.equal(self.prediction, self.label)
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.variable_scope('loss'):
            with tf.variable_scope('softmax_loss'):
                gold_labels = tf.one_hot(indices=self.label, depth=tf.shape(self.logits)[1], name='gold_label')
                softmax_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=gold_labels, logits=self.logits), name='softmax_loss')

            with tf.variable_scope('reg_loss'):
                if self.mode == 'train':
                    tvars = tf.trainable_variables()
                    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg, scope=None)
                    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, tvars)
                    reg_penalty_word_emb = tf.contrib.layers.apply_regularization(l2_regularizer, [self.word_embedding])
                    reg_loss = regularization_penalty - reg_penalty_word_emb
                else:
                    reg_loss = 0

            self.loss = softmax_loss + reg_loss

    def _build_train_op(self):
        with tf.variable_scope('train'):

            self._learning_rate = tf.Variable(0.0, trainable=False, name='learning_rate')

            with tf.variable_scope('optimizer'):

                # clip_weight
                tvars = tf.trainable_variables()
                grads = tf.gradients(self.loss, tvars)
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.args.max_grad_norm)
                grad_var_pairs = zip(grads, tvars)

                if self.args.optim_type == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, epsilon=1e-6, name='adagrad')
                elif self.args.optim_type == 'adam':
                    optimizer = tf.train.AdamOptimizer(self.learning_rate, name='adam')
                elif self.args.optim_type == 'rmsprop':
                    optimizer = tf.train.RMSPropOptimizer(self.learning_rate, name='rmsprop')
                elif self.args.optim_type == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name='sgd')
                else:
                    raise NotImplementedError('Unsupported optimizer: {}'.format(self.args.optim_type))

                self._train_op = optimizer.apply_gradients(grad_var_pairs, name='apply_grad')
                #self._train_op = optimizer.minimize(self.loss)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.learning_rate, lr_value))

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def train_op(self):
        return self._train_op
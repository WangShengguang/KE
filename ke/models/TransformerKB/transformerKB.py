# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''

import tensorflow as tf

from .modules import ff, positional_encoding, multihead_attention
from .._Model import Model


class Transformer(object):
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''

    def __init__(self, num_blocks, num_heads, max_sequence_len, vocab_size,
                 d_model=512, d_ff=2048, dropout_rate=0.0):
        """
        :param num_blocks:
        :param num_heads:
        :param max_sequence_len:
        :param d_model: "hidden dimension of encoder/decoder"
        :param d_ff: "hidden dimension of feedforward layer"
        :param dropout_rate:
        """
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_sequence_len = max_sequence_len
        self.dropout_rate = dropout_rate
        self.d_model = d_model
        self.d_ff = d_ff
        # self.embeddings = get_token_embeddings(vocab_size, self.d_model, zero_pad=True) #todo 出现nan init value
        self.embeddings = tf.get_variable(
            "Tramsformer-W",
            shape=[vocab_size, self.d_model],
            initializer=tf.contrib.layers.xavier_initializer(seed=1234))

    def encode(self, x, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x)  # (N, T1, d_model)
            enc *= self.d_model  # scale

            enc += positional_encoding(enc, self.max_sequence_len)
            enc = tf.layers.dropout(enc, self.dropout_rate, training=training)

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.d_model])
        memory = enc
        return memory


class TransformerKB(Model):
    def __init__(self, data_set, num_ent_tags, num_rel_tags):
        super().__init__(data_set, num_ent_tags, num_rel_tags)
        self.embedding_dim = self.ent_emb_dim
        self.vocab_size = num_ent_tags + num_rel_tags
        self.transformer = Transformer(num_blocks=6, num_heads=8,
                                       max_sequence_len=self.sequence_length,
                                       vocab_size=self.vocab_size, d_model=self.embedding_dim)

    def embedding_def(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        self.total_dims = self.sequence_length * self.embedding_dim
        self.W = tf.get_variable(name="output_W", shape=[self.total_dims, self.num_classes],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1234))
        self.b = tf.Variable(tf.constant(0.0, shape=[self.num_classes]), name="b")

    def forward(self):
        l2_loss = tf.constant(0.0)
        self.encoded = self.transformer.encode(self.hrt_input_x)  # (batch_size, 3, 128)
        self.encoded_flatten = tf.reshape(self.encoded, [-1, self.total_dims])
        self.h_drop = tf.nn.dropout(self.encoded_flatten, self.dropout_keep_prob)
        l2_loss += tf.nn.l2_loss(self.W)
        l2_loss += tf.nn.l2_loss(self.b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, self.W, self.b, name="scores")
        # Calculate loss
        losses = tf.nn.softplus(self.scores * self.input_y)
        self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss

    def predict_def(self):
        self.predict = tf.nn.sigmoid(self.scores, name="predict")

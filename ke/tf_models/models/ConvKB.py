import math

import tensorflow as tf

from ke.config import Config
from ._Model import Model


class ConvKB(Model):
    def __init__(self, data_set, num_ent_tags, num_rel_tags, embedding_size, filter_sizes, num_filters):
        # parms
        self.sequence_length = Config.sequence_len
        self.num_classes = Config.num_classes
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.total_dims = (self.embedding_size * len(filter_sizes) - sum(filter_sizes) + len(
            filter_sizes)) * num_filters
        super().__init__(data_set, num_ent_tags, num_rel_tags)

    def embedding_def(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([num_ent_tags + num_rel_tags, ent_emb_dim], -math.sqrt(1.0 / ent_emb_dim),
                                  math.sqrt(1.0 / ent_emb_dim), seed=1234), name="W")

    def forward(self):
        l2_reg_lambda = 0.001
        dropout_keep_prob = 1.0
        useConstantInit = False
        l2_loss = tf.constant(0.0)
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                if useConstantInit == False:
                    filter_shape = [self.sequence_length, filter_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, seed=1234), name="W")
                else:
                    init1 = tf.constant([[[[0.1]]], [[[0.1]]], [[[-0.1]]]])
                    weight_init = tf.tile(init1, [1, filter_size, 1, self.num_filters])
                    W = tf.get_variable(name="W3", initializer=weight_init)

                b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled_outputs.append(h)

        # Combine all the pooled features
        self.h_pool = tf.concat(pooled_outputs, 2)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.total_dims])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, dropout_keep_prob)

            # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.total_dims, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            b = tf.Variable(tf.constant(0.0, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predict = tf.nn.sigmoid(self.scores, name="predict")
        # Calculate loss
        with tf.name_scope("loss"):
            losses = tf.nn.softplus(self.scores * self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

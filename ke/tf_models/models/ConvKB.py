import math

import tensorflow as tf

from ke.config import Config
from ._model import Model


class ConvKB(Model):

    def __init__(self, filter_sizes, num_filters, vocab_size, data_set,
                 embedding_size=Config.ent_emb_dim, l2_reg_lambda=0.001, dropout_keep_prob=1.0, useConstantInit=False):
        super().__init__(data_set=data_set)
        # parms
        sequence_length = Config.sequence_len
        num_classes = Config.num_classes
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # self.input_x = tf.transpose(self.input_x)  # [[10630,4,1715],[1422,4,18765]] h,r,t
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # [[1],[1],[-1]]
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -math.sqrt(1.0 / embedding_size),
                                                   math.sqrt(1.0 / embedding_size), seed=1234), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                if useConstantInit == False:
                    filter_shape = [sequence_length, filter_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, seed=1234), name="W")
                else:
                    init1 = tf.constant([[[[0.1]]], [[[0.1]]], [[[-0.1]]]])
                    weight_init = tf.tile(init1, [1, filter_size, 1, num_filters])
                    W = tf.get_variable(name="W3", initializer=weight_init)

                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
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
        total_dims = (embedding_size * len(filter_sizes) - sum(filter_sizes) + len(filter_sizes)) * num_filters
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_dims])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, dropout_keep_prob)

            # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[total_dims, num_classes],
                initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predict = tf.nn.sigmoid(self.scores, name="predict")
        # Calculate loss
        with tf.name_scope("loss"):
            losses = tf.nn.softplus(self.scores * self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        # self.saver = tf.train.Saver(max_to_keep=10)  # 写在子类中

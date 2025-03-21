import tensorflow as tf

from ._BaseModel import Model


class ConvKB(Model):
    def __init__(self, data_set, num_ent_tags, num_rel_tags):
        super().__init__(data_set, num_ent_tags, num_rel_tags)
        self.filter_sizes = [1]
        self.num_filters = 500
        self.useConstantInit = False

    def forward(self):
        l2_loss = tf.constant(0.0)
        self.embedded_chars_expanded = tf.expand_dims(self.hrt_embed, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                if self.useConstantInit == False:
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
        total_dims = (self.common_emb_dim * len(self.filter_sizes) - sum(self.filter_sizes) + len(
            self.filter_sizes)) * self.num_filters
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_dims])

        # Add dropout
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[total_dims, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            b = tf.Variable(tf.constant(0.0, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        # Calculate loss
        losses = tf.nn.softplus(self.scores * self.input_y)
        self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss

    def predict_def(self):
        self.predict = tf.nn.sigmoid(self.scores, name="predict")

import tensorflow as tf

from .modules import ff, positional_encoding, multihead_attention
from .._BaseModel import Model


class TransformerKB(Model):
    def __init__(self, data_set, num_ent_tags, num_rel_tags):
        super().__init__(data_set, num_ent_tags, num_rel_tags)
        self.num_blocks = 3
        self.num_heads = 4
        self.dropout_rate = 1 - self.config.dropout_keep_prob
        self.d_ff = 2048

    def transformer_encode(self, enc, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        # embedding
        enc *= self.common_emb_dim  # scale

        enc += positional_encoding(enc, self.sequence_length)

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
                enc = ff(enc, num_units=[self.d_ff, self.common_emb_dim])
        memory = enc
        return memory

    def forward(self):
        self.encoded = self.transformer_encode(self.hrt_embed)  # (batch_size, 3, 128)
        total_dims = self.sequence_length * self.common_emb_dim
        self.encoded_flatten = tf.reshape(self.encoded, [-1, total_dims])
        #
        # self.encoded_flatten = tf.nn.relu(self.encoded_flatten)
        self.h_drop = tf.nn.dropout(self.encoded_flatten, self.dropout_keep_prob)
        # output
        self.W = tf.get_variable(name="output_W", shape=[total_dims, self.num_classes],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1234))
        self.b = tf.Variable(tf.constant(0.0, shape=[self.num_classes]), name="b")

        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(self.W)
        l2_loss += tf.nn.l2_loss(self.b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, self.W, self.b, name="scores")
        # Calculate loss
        losses = tf.nn.softplus(self.scores * self.input_y)
        self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss

    def predict_def(self):
        self.predict = tf.nn.sigmoid(self.scores, name="predict")

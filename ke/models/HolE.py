import tensorflow as tf

from ._BaseModel import TransX


class HolE(TransX):
    r'''
    HolE employs circular correlations to create compositional representations.
    HolE can capture rich interactions but simultaneously remains efficient to compute.
    '''

    def _cconv(self, a, b):
        return tf.ifft(tf.fft(a) * tf.fft(b)).real

    def _ccorr(self, a, b):
        a = tf.cast(a, tf.complex64)
        b = tf.cast(b, tf.complex64)
        return tf.real(tf.ifft(tf.conj(tf.fft(a)) * tf.fft(b)))

    def _calc(self, head, tail, rel, name=None):
        relation_mention = tf.nn.l2_normalize(rel, 1)
        entity_mention = self._ccorr(head, tail)
        return -tf.sigmoid(tf.reduce_sum(relation_mention * entity_mention, 1, keep_dims=True), name=name)

    def forward(self):
        _p_score = tf.reshape(self._calc(self.pos_h_embed, self.pos_t_embed, self.pos_r_embed), [-1, 1])
        _n_score = tf.reshape(self._calc(self.neg_h_embed, self.neg_t_embed, self.neg_r_embed), [-1, 1])
        p_score = _p_score
        n_score = tf.reduce_mean(_n_score, 1, keep_dims=True)
        # Calculating loss to get what the framework will optimize
        self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + self.config.margin, 0))

    def predict_def(self):
        self.predict = tf.reduce_sum(self._calc(self.h_embed, self.t_embed, self.r_embed), 1, keep_dims=True,
                                     name="predict")

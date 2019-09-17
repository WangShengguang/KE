import tensorflow as tf

from config import Config
from ._Model import TransX


class TransE(TransX):
    r'''
    TransE is the first model to introduce translation-based embedding,
    which interprets relations as the translations operating on entities.
    '''

    def _calc(self, h, t, r):
        h = tf.nn.l2_normalize(h, -1)
        t = tf.nn.l2_normalize(t, -1)
        r = tf.nn.l2_normalize(r, -1)
        return tf.abs(h + r - t)

    def forward(self):
        _p_score = self._calc(self.pos_h_embed, self.pos_t_embed, self.pos_r_embed)
        _n_score = self._calc(self.neg_h_embed, self.neg_t_embed, self.neg_r_embed)
        # The shape of p_score is (batch_size, 1, 1)
        # The shape of n_score is (batch_size, negative_ent + negative_rel, 1)
        self.p_score = tf.reduce_sum(_p_score, -1, keep_dims=True)
        self.n_score = tf.reduce_sum(_n_score, -1, keep_dims=True)
        # Calculating loss to get what the framework will optimize
        self.loss = tf.reduce_mean(tf.maximum(self.p_score - self.n_score + Config.margin, 0), name="loss")

    def predict_def(self):
        self.predict = tf.reduce_mean(self._calc(self.h_embed, self.t_embed, self.r_embed),
                                      -1, keep_dims=False, name="predict")

import tensorflow as tf

from ._BaseModel import TransX


class DistMult(TransX):
    r'''
    DistMult is based on the bilinear model where each relation is represented by a diagonal rather than a full matrix.
    DistMult enjoys the same scalable property as TransE and it achieves superior performance over TransE.
    '''

    def _calc(self, h, t, r):
        return tf.reduce_sum(h * r * t, -1, keep_dims=False)

    def forward(self):
        _p_score = self._calc(self.pos_h_embed, self.pos_t_embed, self.pos_r_embed)
        _n_score = self._calc(self.neg_h_embed, self.neg_t_embed, self.neg_r_embed)
        # print(_n_score.get_shape())
        loss_func = tf.reduce_mean(tf.nn.softplus(- self.pos_y * _p_score) + tf.nn.softplus(- self.neg_y * _n_score))
        regul_func = tf.reduce_mean(self.pos_h_embed ** 2 + self.pos_t_embed ** 2 + self.pos_r_embed ** 2 +
                                    self.neg_h_embed ** 2 + self.neg_t_embed ** 2 + self.neg_r_embed ** 2)
        self.loss = loss_func + self.config.l2_reg_lambda * regul_func

    def predict_def(self):
        self.predict = tf.multiply(self._calc(self.h_embed, self.t_embed, self.r_embed), -1, name="predict")

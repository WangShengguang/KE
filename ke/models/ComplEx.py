import tensorflow as tf

from ._BaseModel import TransX


class ComplEx(TransX):
    r'''
    ComplEx extends DistMult by introducing complex-valued embeddings so as to better model asymmetric relations.
    It is proved that HolE is subsumed by ComplEx as a special case.
    '''

    def embedding_def(self):
        self.ent_embeddings_2 = tf.get_variable(name="ent_embeddings_2", shape=[self.num_ent_tags, self.ent_emb_dim],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings_2 = tf.get_variable(name="rel_embeddings_2", shape=[self.num_rel_tags, self.rel_emb_dim],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def _calc(self, h_embed_1, h_embed_2, t_embed_1, t_embed_2, r_embed_1, r_embed_2, name=None):
        return tf.reduce_sum(h_embed_1 * t_embed_1 * r_embed_1 + h_embed_2 * t_embed_2 * r_embed_1 +
                             h_embed_1 * t_embed_2 * r_embed_2 - h_embed_2 * t_embed_1 * r_embed_2,
                             -1, keep_dims=False, name=name)

    def forward(self):
        pos_h_embed_2 = tf.nn.embedding_lookup(self.ent_embeddings_2, self.pos_h)
        pos_t_embed_2 = tf.nn.embedding_lookup(self.ent_embeddings_2, self.pos_t)
        pos_r_embed_2 = tf.nn.embedding_lookup(self.rel_embeddings_2, self.pos_r)

        neg_h_embed_2 = tf.nn.embedding_lookup(self.ent_embeddings_2, self.neg_h)
        neg_t_embed_2 = tf.nn.embedding_lookup(self.ent_embeddings_2, self.neg_t)
        neg_r_embed_2 = tf.nn.embedding_lookup(self.rel_embeddings_2, self.neg_r)

        _p_score = self._calc(self.pos_h_embed, pos_h_embed_2, self.pos_t_embed, pos_t_embed_2,
                              self.pos_r_embed, pos_r_embed_2)
        _n_score = self._calc(self.neg_h_embed, neg_h_embed_2, self.neg_t_embed, neg_t_embed_2,
                              self.neg_r_embed, neg_r_embed_2)

        loss_func = tf.reduce_mean(tf.nn.softplus(- self.pos_y * _p_score) + tf.nn.softplus(- self.neg_y * _n_score))
        regul_func = tf.reduce_mean(self.pos_h_embed ** 2 + self.pos_t_embed ** 2 + self.pos_r_embed ** 2 +
                                    self.neg_h_embed ** 2 + self.neg_t_embed ** 2 + self.neg_r_embed ** 2 +
                                    pos_h_embed_2 ** 2 + pos_t_embed_2 ** 2 + pos_r_embed_2 ** 2 +
                                    neg_h_embed_2 ** 2 + neg_t_embed_2 ** 2 + neg_r_embed_2 ** 2)
        self.loss = loss_func + self.config.l2_reg_lambda * regul_func

    def predict_def(self):
        h_embed_2 = tf.nn.embedding_lookup(self.ent_embeddings_2, self.h)
        t_embed_2 = tf.nn.embedding_lookup(self.ent_embeddings_2, self.t)
        r_embed_2 = tf.nn.embedding_lookup(self.rel_embeddings_2, self.r)
        self.predict = tf.multiply(self._calc(self.h_embed, h_embed_2, self.t_embed, t_embed_2,
                                              self.r_embed, r_embed_2), -1, name="predict")

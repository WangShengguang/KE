import tensorflow as tf

from ._Model import TransX


class TransR(TransX):
    r'''
    TransR first projects entities from entity space to corresponding relation space
    and then builds translations between projected entities.
    '''

    def _transfer(self, transfer_matrix, embeddings):
        return tf.matmul(embeddings, transfer_matrix)

    def _calc(self, h, t, r):
        h = tf.nn.l2_normalize(h, -1)
        t = tf.nn.l2_normalize(t, -1)
        r = tf.nn.l2_normalize(r, -1)
        return abs(h + r - t)

    def embedding_def(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        self.transfer_matrix = tf.get_variable(name="transfer_matrix",
                                               shape=[num_rel_tags, ent_emb_dim * rel_emb_dim],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def forward(self):
        # Getting the required mapping matrices
        pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, self.pos_r),
                                [-1, self.ent_emb_dim, self.rel_emb_dim])
        # Calculating score functions for all positive triples and negative triples
        p_h = self._transfer(pos_matrix, self.pos_h_embed)
        p_t = self._transfer(pos_matrix, self.pos_t_embed)
        p_r = self.pos_r_embed
        negative_rel = 0
        if negative_rel == 0:  # config.negative_rel == 0: 关系负例个数为 0，当前默认
            n_h = self._transfer(pos_matrix, self.neg_h_embed)
            n_t = self._transfer(pos_matrix, self.neg_t_embed)
            n_r = self.neg_r_embed
        else:
            neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, self.neg_r),
                                    [-1, self.ent_emb_dim, self.rel_emb_dim])
            n_h = self._transfer(neg_matrix, self.neg_h_embed)
            n_t = self._transfer(neg_matrix, self.neg_t_embed)
            n_r = self.neg_r_embed
        # The shape of _p_score is (batch_size, 1, hidden_size)
        # The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
        _p_score = self._calc(p_h, p_t, p_r)
        _n_score = self._calc(n_h, n_t, n_r)
        # The shape of p_score is (batch_size, 1, 1)
        # The shape of n_score is (batch_size, negative_ent + negative_rel, 1)
        p_score = tf.reduce_sum(_p_score, -1, keep_dims=True)
        n_score = tf.reduce_sum(_n_score, -1, keep_dims=True)
        # Calculating loss to get what the framework will optimize
        self.loss = tf.reduce_mean(tf.maximum(p_score - n_score + self.config.margin, 0))

    def predict_def(self):
        predict_h_e = tf.reshape(self.h_embed, [1, -1, self.ent_emb_dim])
        predict_t_e = tf.reshape(self.t_embed, [1, -1, self.ent_emb_dim])
        predict_r_e = tf.reshape(self.r_embed, [1, -1, self.rel_emb_dim])
        predict_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, self.r[0]),
                                    [1, self.ent_emb_dim, self.rel_emb_dim])
        h_e = tf.reshape(self._transfer(predict_matrix, predict_h_e), [-1, self.rel_emb_dim])
        t_e = tf.reshape(self._transfer(predict_matrix, predict_t_e), [-1, self.rel_emb_dim])
        r_e = predict_r_e
        self.predict = tf.reduce_sum(self._calc(h_e, t_e, r_e), -1, keep_dims=True, name="predict")

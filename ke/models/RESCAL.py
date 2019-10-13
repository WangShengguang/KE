import tensorflow as tf

from ._BaseModel import TransX


class RESCAL(TransX):
    r'''
    RESCAL is a tensor factorization approach to knowledge representation learning,
    which is able to perform collective learning via the latent components of the factorization.
    '''

    def embedding_def(self):
        self.rel_matrices = tf.get_variable(name="rel_matrices",
                                            shape=[self.num_rel_tags, self.common_emb_dim * self.common_emb_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def _calc(self, h, t, r):
        return h * tf.matmul(r, t)

    def forward(self):
        # Embedding entities and relations of triples, e.g. p_h, p_t and p_r are embeddings for positive triples
        p_h = tf.reshape(self.pos_h_embed, [-1, self.common_emb_dim, 1])
        p_t = tf.reshape(self.pos_t_embed, [-1, self.common_emb_dim, 1])
        p_r = tf.reshape(tf.nn.embedding_lookup(self.rel_matrices, self.pos_r),
                         [-1, self.common_emb_dim, self.common_emb_dim])
        #
        n_h = tf.reshape(self.neg_h_embed, [-1, self.common_emb_dim, 1])
        n_t = tf.reshape(self.neg_t_embed, [-1, self.common_emb_dim, 1])
        n_r = tf.reshape(tf.nn.embedding_lookup(self.rel_matrices, self.neg_r),
                         [-1, self.common_emb_dim, self.common_emb_dim])

        _p_score = tf.reshape(self._calc(p_h, p_t, p_r), [-1, 1, self.common_emb_dim])
        _n_score = tf.reshape(self._calc(n_h, n_t, n_r), [-1, 1, self.common_emb_dim])

        p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims=False), 1, keep_dims=True)
        n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims=False), 1, keep_dims=True)

        # Calculating loss to get what the framework will optimize
        self.loss = tf.reduce_sum(tf.maximum(n_score - p_score + self.config.margin, 0))

    def predict_def(self):
        h = tf.reshape(self.h_embed, [-1, self.common_emb_dim, 1])
        t = tf.reshape(self.t_embed, [-1, self.common_emb_dim, 1])
        r = tf.reshape(tf.nn.embedding_lookup(self.rel_matrices, self.r),
                       [-1, self.common_emb_dim, self.common_emb_dim])
        self.predict = tf.reduce_sum(-self._calc(h, t, r), 1, keep_dims=False, name="predict")

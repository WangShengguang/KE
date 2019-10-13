import tensorflow as tf

from ._BaseModel import TransX


class TransH(TransX):
    r'''
    To preserve the mapping propertities of 1-N/N-1/N-N relations,
    TransH inperprets a relation as a translating operation on a hyperplane.
    '''

    def _transfer(self, e, n):
        n = tf.nn.l2_normalize(n, -1)
        return e - tf.reduce_sum(e * n, -1, keepdims=True) * n

    def _calc(self, h, t, r):
        h = tf.nn.l2_normalize(h, -1)
        t = tf.nn.l2_normalize(t, -1)
        r = tf.nn.l2_normalize(r, -1)
        return abs(h + r - t)

    def embedding_def(self):
        self.normal_vectors = tf.get_variable(name="normal_vectors", shape=[self.num_rel_tags, self.rel_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def forward(self):
        # Getting the required normal vectors of planes to transfer entity embeddings
        pos_norm = tf.nn.embedding_lookup(self.normal_vectors, self.pos_r)
        neg_norm = tf.nn.embedding_lookup(self.normal_vectors, self.neg_r)
        # Calculating score functions for all positive triples and negative triples
        p_h = self._transfer(self.pos_h_embed, pos_norm)
        p_t = self._transfer(self.pos_t_embed, pos_norm)
        p_r = self.pos_r_embed
        n_h = self._transfer(self.neg_h_embed, neg_norm)
        n_t = self._transfer(self.neg_t_embed, neg_norm)
        n_r = self.neg_r_embed
        # Calculating score functions for all positive triples and negative triples
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
        predict_norm = tf.nn.embedding_lookup(self.normal_vectors, self.r)
        h_e = self._transfer(self.h_embed, predict_norm)
        t_e = self._transfer(self.t_embed, predict_norm)
        r_e = self.r_embed
        self.predict = tf.reduce_sum(self._calc(h_e, t_e, r_e), -1, keepdims=True, name="predict")

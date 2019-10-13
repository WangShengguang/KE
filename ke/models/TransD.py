import tensorflow as tf

from ._BaseModel import TransX


def tf_resize(tensor, axis, size):
    shape = tensor.get_shape().as_list()
    osize = shape[axis]
    if osize == size:
        return tensor
    if (osize > size):
        shape[axis] = size
        return tf.slice(tensor, begin=(0,) * len(shape), size=shape)
    paddings = [[0, 0] for i in range(len(shape))]
    paddings[axis][1] = size - osize
    return tf.pad(tensor, paddings=paddings)


class TransD(TransX):
    r'''
    TransD constructs a dynamic mapping matrix for each entity-relation pair by considering the diversity of entities and relations simultaneously.
    Compared with TransR/CTransR, TransD has fewer parameters and has no matrix vector multiplication.
    '''

    def _transfer(self, e, t, r):
        # return e + tf.reduce_sum(e * t, -1, keep_dims = True) * r
        return tf_resize(e, -1, r.get_shape()[-1]) + tf.reduce_sum(e * t, -1, keepdims=True) * r

    def _calc(self, h, t, r):
        h = tf.nn.l2_normalize(h, -1)
        t = tf.nn.l2_normalize(t, -1)
        r = tf.nn.l2_normalize(r, -1)
        return abs(h + r - t)

    def embedding_def(self):
        self.ent_transfer = tf.get_variable(name="ent_transfer", shape=[self.num_ent_tags, self.ent_emb_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_transfer = tf.get_variable(name="rel_transfer", shape=[self.num_rel_tags, self.rel_emb_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def forward(self):
        # Getting the required parameters to transfer entity embeddings, e.g. pos_h_t, pos_t_t and pos_r_t are transfer parameters for positive triples
        pos_h_transfer_embed = tf.nn.embedding_lookup(self.ent_transfer, self.pos_h)
        pos_t_transfer_embed = tf.nn.embedding_lookup(self.ent_transfer, self.pos_t)
        pos_r_transfer_embed = tf.nn.embedding_lookup(self.rel_transfer, self.pos_r)
        neg_h_transfer_embed = tf.nn.embedding_lookup(self.ent_transfer, self.neg_h)
        neg_t_transfer_embed = tf.nn.embedding_lookup(self.ent_transfer, self.neg_t)
        neg_r_transfer_embed = tf.nn.embedding_lookup(self.rel_transfer, self.neg_r)

        # Calculating score functions for all positive triples and negative triples
        p_h = self._transfer(self.pos_h_embed, pos_h_transfer_embed, pos_r_transfer_embed)
        p_t = self._transfer(self.pos_t_embed, pos_t_transfer_embed, pos_r_transfer_embed)
        p_r = self.pos_r_embed
        n_h = self._transfer(self.neg_h_embed, neg_h_transfer_embed, neg_r_transfer_embed)
        n_t = self._transfer(self.neg_t_embed, neg_t_transfer_embed, neg_r_transfer_embed)
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
        h_transfer_embed = tf.nn.embedding_lookup(self.ent_transfer, self.h)
        t_transfer_embed = tf.nn.embedding_lookup(self.ent_transfer, self.t)
        r_transfer_embed = tf.nn.embedding_lookup(self.rel_transfer, self.r)
        h_e = self._transfer(self.h_embed, h_transfer_embed, r_transfer_embed)
        t_e = self._transfer(self.t_embed, t_transfer_embed, r_transfer_embed)
        r_e = self.r_embed
        self.predict = tf.reduce_sum(self._calc(h_e, t_e, r_e), -1, keep_dims=True, name="predict")

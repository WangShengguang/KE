import tensorflow as tf

from config import Config
from ._Model import Model


class TransE(Model):
    r'''
    TransE is the first model to introduce translation-based embedding,
    which interprets relations as the translations operating on entities.
    '''

    def embedding_def(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[num_ent_tags, ent_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[num_rel_tags, rel_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def _calc(self, h, t, r):
        h = tf.nn.l2_normalize(h, -1)
        t = tf.nn.l2_normalize(t, -1)
        r = tf.nn.l2_normalize(r, -1)
        return tf.abs(h + r - t)

    def forward(self):
        # Obtaining the initial configuration of the model
        # The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
        # The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
        p_h = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
        p_t = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
        p_r = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
        n_h = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
        n_t = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
        n_r = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
        h = tf.nn.embedding_lookup(self.ent_embeddings, self.h)
        t = tf.nn.embedding_lookup(self.ent_embeddings, self.t)
        r = tf.nn.embedding_lookup(self.rel_embeddings, self.r)
        # Calculating score functions for all positive triples and negative triples
        # The shape of _p_score is (batch_size, 1, hidden_size)
        # The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
        _p_score = self._calc(p_h, p_t, p_r)
        _n_score = self._calc(n_h, n_t, n_r)
        # The shape of p_score is (batch_size, 1, 1)
        # The shape of n_score is (batch_size, negative_ent + negative_rel, 1)
        self.p_score = tf.reduce_sum(_p_score, -1, keep_dims=True)
        self.n_score = tf.reduce_sum(_n_score, -1, keep_dims=True)
        self.predict = tf.reduce_mean(self._calc(h, t, r), -1, keep_dims=False, name="predict")
        # Calculating loss to get what the framework will optimize
        self.loss = tf.reduce_mean(tf.maximum(self.p_score - self.n_score + Config.margin, 0), name="loss")

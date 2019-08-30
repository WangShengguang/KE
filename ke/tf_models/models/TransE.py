import tensorflow as tf

from ke.config import Config
from ._model import TransXModel


class TransE(TransXModel):
    r'''
    TransE is the first model to introduce translation-based embedding,
    which interprets relations as the translations operating on entities.
    '''

    def __init__(self, num_ent_tags, num_rel_tags, data_set, ent_emb_dim=Config.ent_emb_dim,
                 rel_emb_dim=Config.rel_emb_dim):
        super().__init__(data_set=data_set)
        # Obtaining the initial configuration of the model
        # Defining required parameters of the model, including embeddings of entities and relations
        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[num_ent_tags, ent_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[num_rel_tags, rel_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.parameter_lists = {"ent_embeddings": self.ent_embeddings,
                                "rel_embeddings": self.rel_embeddings}
        self.loss_def()
        self.predict_def()

    def _calc(self, h, t, r):
        h = tf.nn.l2_normalize(h, -1)
        t = tf.nn.l2_normalize(t, -1)
        r = tf.nn.l2_normalize(r, -1)
        return tf.abs(h + r - t)

    def loss_def(self):
        # Obtaining the initial configuration of the model
        # The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
        # The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
        p_h = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
        p_t = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
        p_r = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
        n_h = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
        n_t = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
        n_r = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
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
        self.loss = tf.reduce_mean(tf.maximum(p_score - n_score + Config.margin, 0), name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def predict_def(self):
        predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.predict_h)
        predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.predict_t)
        predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.predict_r)
        self.predict = tf.reduce_mean(self._calc(predict_h_e, predict_t_e, predict_r_e), 1, keep_dims=False,
                                      name="predict")

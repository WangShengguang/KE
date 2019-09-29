import tensorflow as tf

from ._BaseModel import Model


class Analogy(Model):

    def embedding_def(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        # embedding def
        self.ent_embeddings_1 = tf.get_variable(name="ent_embeddings_1", shape=[num_ent_tags, ent_emb_dim // 2],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings_1 = tf.get_variable(name="rel_embeddings_1", shape=[num_rel_tags, rel_emb_dim // 2],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.ent_embeddings_2 = tf.get_variable(name="ent_embeddings_2", shape=[num_ent_tags, ent_emb_dim // 2],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings_2 = tf.get_variable(name="rel_embeddings_2", shape=[num_rel_tags, rel_emb_dim // 2],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[num_ent_tags, ent_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[num_rel_tags, rel_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    # score function for ComplEx
    def _calc_comp(self, e1_h, e2_h, e1_t, e2_t, r1, r2):
        return e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2

    # score function for DistMult
    def _calc_dist(self, e_h, e_t, rel):
        return e_h * e_t * rel

    def forward(self):
        # Embedding entities and relations of triples
        self.h_embed_1 = tf.nn.embedding_lookup(self.ent_embeddings_1, self.h)
        self.t_embed_1 = tf.nn.embedding_lookup(self.ent_embeddings_1, self.t)
        self.r_embed_1 = tf.nn.embedding_lookup(self.rel_embeddings_1, self.r)

        self.h_embed_2 = tf.nn.embedding_lookup(self.ent_embeddings_2, self.h)
        self.t_embed_2 = tf.nn.embedding_lookup(self.ent_embeddings_2, self.t)
        self.r_embed_2 = tf.nn.embedding_lookup(self.rel_embeddings_2, self.r)

        self.h_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.h)
        self.t_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.t)
        self.r_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.r)

        # Calculating score functions for all positive triples and negative triples
        loss_func = tf.reduce_mean(tf.nn.softplus(tf.reshape(self.input_y, [-1]) * tf.reshape(self.predict, [-1])),
                                   0, keep_dims=False)
        regul_func = tf.reduce_mean(self.h_embed_1 ** 2) + tf.reduce_mean(self.t_embed_1 ** 2) + tf.reduce_mean(
            self.h_embed_2 ** 2) + tf.reduce_mean(
            self.t_embed_2 ** 2) + tf.reduce_mean(self.r_embed_1 ** 2) + tf.reduce_mean(
            self.r_embed_2 ** 2) + tf.reduce_mean(
            self.h_embed ** 2) + tf.reduce_mean(
            self.t_embed ** 2) + tf.reduce_mean(self.r_embed ** 2)
        # Calculating loss to get what the framework will optimize
        self.loss = loss_func + self.config.l2_reg_lambda * regul_func

    def predict_def(self):
        res_comp = -tf.reduce_sum(
            self._calc_comp(self.h_embed_1, self.h_embed_2, self.t_embed_1, self.t_embed_2, self.r_embed_1,
                            self.r_embed_2), 1, keep_dims=False)
        res_dist = -tf.reduce_sum(self._calc_dist(self.h_embed, self.t_embed, self.r_embed), 1, keep_dims=False)
        self.predict = tf.add(res_comp, res_dist, name="predict")

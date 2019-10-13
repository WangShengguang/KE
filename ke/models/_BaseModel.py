from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import Tensor, Operation

from config import Config
from ke.utils.saver import Saver

__all__ = ["Model"]


class Model(ABC):
    """  网络构建必须在 __network_build函数调用的几个函数中实现
        实现 model.save 和 model.load
        https://zhuanlan.zhihu.com/p/68899384
        https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
    """

    def __init__(self, data_set=None, num_ent_tags=None, num_rel_tags=None,
                 ent_emb_dim=Config.ent_emb_dim, rel_emb_dim=Config.rel_emb_dim, common_emb_dim=Config.common_emb_dim,
                 sequence_length=Config.sequence_len, batch_size=Config.batch_size, num_classes=Config.num_classes,
                 name=None):
        """ 为统一训练方式，几个属性张量必须在子类实现
        input_x,input_y,loss,train_op,predict
        :param data_set: 数据集名称，用来作为模型保存的相对目录
                model_path = checkpoint_dir/data_set/model_name  *.meta
        """
        # for model save & load
        self.name = name if name else self.__class__.__name__  # model name
        self.data_set = data_set
        self.config = Config()
        # params
        self.num_ent_tags = num_ent_tags
        self.num_rel_tags = num_rel_tags
        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim
        self.common_emb_dim = common_emb_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        #
        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [self.batch_size, self.num_classes], name="input_y")  # [[1],[1],[-1]]
        self.h, self.t, self.r = tf.unstack(self.input_x, axis=1)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def _build(self):
        """子类 __init__完成之后再调用此方法,通过元类实现
            input_x: [[10630,1715,4],[1422,18765,4]] h,t,r
        """
        self.input_def()  # placeholder
        self.embedding_def()  # weights
        self.forward()  # forward propagate
        self.predict_def()
        self.backward()  # backward propagate
        self.saver = Saver(data_set=self.data_set, model_name=self.name)
        assert isinstance(self.loss, Tensor), "please set attr loss for backward"
        assert isinstance(self.predict, Tensor), "please set attr predict for predict"
        assert isinstance(self.train_op, Operation), "please set attr train_op for backward"
        assert self.predict.name == "predict:0", "predict variable must be set name 'predict'"
        # self.predict = tf.add(a, b, name="predict")

    def input_def(self):
        """
        https://stackoverflow.com/questions/44576167/force-child-class-to-override-parents-methods
        https://stackoverflow.com/questions/3948873/prevent-function-overriding-in-python
        """
        # self.hrt_input_x = tf.stack([self.h, self.r, self.t], axis=1)  # 交换了位置 htr->hrt
        # self.common_embeddings = tf.get_variable([self.ent_embeddings, self.rel_embeddings], axix=0,
        #                                       name="common_embeddings")
        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[self.num_ent_tags, self.common_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[self.num_rel_tags, self.common_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        self.h_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.h)
        self.t_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.t)
        self.r_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.r)

        self.hrt_embed = tf.stack([self.h_embed, self.r_embed, self.t_embed], axis=1)

    def embedding_def(self):
        pass

    @abstractmethod
    def forward(self):
        self.loss = None
        self.predict = None
        raise NotImplementedError

    def predict_def(self):
        pass

    def backward(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return self.train_op


class TransX(Model):

    def input_def(self):
        positive_indices = tf.where(tf.greater(tf.reshape(self.input_y, [-1]), 0.999))
        negative_indices = tf.where(tf.less(tf.reshape(self.input_y, [-1]), -0.999))  # negative sample label 0 or 1
        self.pos_y = tf.ones_like(positive_indices, dtype=tf.float32)
        self.neg_y = -1.0 * tf.ones_like(negative_indices, dtype=tf.float32)
        positive_samples = tf.reshape(tf.gather_nd(self.input_x, positive_indices),
                                      [self.batch_size // 2, self.sequence_length])
        negative_samples = tf.reshape(tf.gather_nd(self.input_x, negative_indices),
                                      [self.batch_size // 2, self.sequence_length])
        self.pos_h, self.pos_t, self.pos_r = tf.unstack(positive_samples, axis=1)
        self.neg_h, self.neg_t, self.neg_r = tf.unstack(negative_samples, axis=1)
        # embedding def
        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[self.num_ent_tags, self.ent_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[self.num_rel_tags, self.rel_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        self.pos_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
        self.pos_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
        self.pos_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
        self.neg_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
        self.neg_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
        self.neg_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
        self.h_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.h)
        self.t_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.t)
        self.r_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.r)

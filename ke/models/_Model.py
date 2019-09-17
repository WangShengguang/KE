import tensorflow as tf

from config import Config
from ke.utils.saver import Saver

__all__ = ["Model"]


class NetworkMeta(type):

    def __new__(mcs, class_name, class_parents, class_attr):
        if class_name not in __all__:  # subclass
            _parent_init = class_parents[0].__init__
            if "__init__" in class_attr:
                _cur_init = class_attr["__init__"]  # subclass __init__

                def _init(self, *args, **kwargs):
                    # _parent_init(self, *args, **kwargs) #子类init中一般已经完成对父类init的调用
                    _cur_init(self, *args, **kwargs)
                    getattr(self, "_Model__build_network")()  # 子类初始化完成后调用_network_build
            else:

                def _init(self, *args, **kwargs):
                    _parent_init(self, *args, **kwargs)
                    getattr(self, "_Model__build_network")()  # 子类初始化完成后调用_network_build

            class_attr["__init__"] = _init

        return type.__new__(mcs, class_name, class_parents, class_attr)


class Model(object):
    """  网络构建必须在 __network_build函数调用的几个函数中实现
        实现 model.save 和 model.load
        https://zhuanlan.zhihu.com/p/68899384
        https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
    """

    def __init__(self, data_set=None, num_ent_tags=None, num_rel_tags=None,
                 ent_emb_dim=Config.ent_emb_dim, rel_emb_dim=Config.rel_emb_dim,
                 sequence_length=Config.sequence_len, batch_size=Config.batch_size, num_classes=Config.num_classes,
                 name=None, checkpoint_dir=Config.tf_ckpt_dir):
        """ 为统一训练方式，几个属性张量必须在子类实现
        input_x,input_y,loss,train_op,predict
        :param data_set: 数据集名称，用来作为模型保存的相对目录
                model_path = checkpoint_dir/data_set/model_name  *.meta
        """
        # for model save & load
        self.name = name if name else self.__class__.__name__  # model name
        self.checkpoint_dir = checkpoint_dir
        self.data_set = data_set
        self.config = Config()
        # params
        self.num_ent_tags = num_ent_tags
        self.num_rel_tags = num_rel_tags
        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        # build model
        # self._build(num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim, **kwargs)

    def _build(self):
        """子类 __init__完成之后再调用此方法,通过元类实现"""
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.input_def()  # placeholder
        self.embedding_def(self.num_ent_tags, self.num_rel_tags, self.ent_emb_dim, self.rel_emb_dim)  # weights
        self.forward()  # forward propagate
        self.predict_def()
        self.backward()  # backward propagate
        self.saver = Saver(max_to_keep=Config.max_to_keep, model_name=self.name,
                           checkpoint_dir=self.checkpoint_dir, relative_dir=self.data_set)
        # tf.global_variables()  # TODO input_x,input_y,loss,train_op,predict
        assert getattr(self, "loss") is not None, "please set_attr loss for backforward"
        assert getattr(self, "predict") is not None, "please set_attr predict for predict"

    def input_def(self):
        sequence_length = self.sequence_length
        num_classes = self.num_classes
        batch_size = self.batch_size
        # [[10630,4,1715],[1422,4,18765]] h,r,t
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")  # [[1],[1],[-1]]
        self.h, self.t, self.r = tf.unstack(self.input_x, axis=1)
        self.hrt_input_x = tf.stack([self.h, self.r, self.t], axis=1)  # 交换了位置

    def embedding_def(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        pass

    def forward(self):
        self.loss = None
        self.predict = None
        raise NotImplementedError

    def predict_def(self):
        pass

    def backward(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return self.train_op


class TransX(Model):
    """  网络构建必须在 __network_build函数调用的几个函数中实现
        实现 model.save 和 model.load
        https://zhuanlan.zhihu.com/p/68899384
        https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
    """

    def __init__(self, data_set=None, num_ent_tags=None, num_rel_tags=None,
                 ent_emb_dim=Config.ent_emb_dim, rel_emb_dim=Config.rel_emb_dim,
                 sequence_length=Config.sequence_len, batch_size=Config.batch_size, num_classes=Config.num_classes,
                 name=None, checkpoint_dir=Config.tf_ckpt_dir):
        """ 为统一训练方式，几个属性张量必须在子类实现
        input_x,input_y,loss,train_op,predict
        :param data_set: 数据集名称，用来作为模型保存的相对目录
                model_path = checkpoint_dir/data_set/model_name  *.meta
        """
        # for model save & load
        super().__init__(data_set=data_set, num_ent_tags=num_ent_tags, num_rel_tags=num_rel_tags,
                         ent_emb_dim=ent_emb_dim, rel_emb_dim=rel_emb_dim,
                         sequence_length=sequence_length, batch_size=batch_size, num_classes=num_classes,
                         name=name, checkpoint_dir=checkpoint_dir)

    def input_def(self):
        sequence_length = self.sequence_length
        num_classes = self.num_classes
        batch_size = self.batch_size
        # input def  [[10630,4,1715],[1422,4,18765]] h,r,t
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")  # [[1],[1],[-1]]
        self.h, self.t, self.r = tf.unstack(self.input_x, axis=1)
        positive_indices = tf.where(tf.greater(tf.reshape(self.input_y, [-1]), 0.999))
        negative_indices = tf.where(tf.less(tf.reshape(self.input_y, [-1]), 0.001))  # negative sample label 0 or 1
        self.positive_samples = tf.reshape(tf.gather_nd(self.input_x, positive_indices),
                                           [batch_size // 2, sequence_length])
        self.negative_samples = tf.reshape(tf.gather_nd(self.input_x, negative_indices),
                                           [batch_size // 2, sequence_length])
        self.pos_h, self.pos_t, self.pos_r = tf.unstack(self.positive_samples, axis=1)
        self.neg_h, self.neg_t, self.neg_r = tf.unstack(self.negative_samples, axis=1)
        # embedding def
        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[self.num_ent_tags, self.ent_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[self.num_rel_tags, self.rel_emb_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.pos_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
        self.pos_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
        self.pos_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
        self.neg_h_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
        self.neg_t_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
        self.neg_r_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
        self.h_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.h)
        self.t_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.t)
        self.r_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.r)

    def embedding_def(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        pass

    def forward(self):
        self.loss = None
        self.predict = None
        raise NotImplementedError

    def backward(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return self.train_op

import tensorflow as tf

from ke.config import Config
from ke.tf_models.model_utils.saver import Saver


class Model(object):
    """ 轻度封装，实现 model.save 和 model.load
        https://zhuanlan.zhihu.com/p/68899384
        https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
    """

    def __init__(self, data_set, num_ent_tags, num_rel_tags,
                 ent_emb_dim=Config.ent_emb_dim, rel_emb_dim=Config.rel_emb_dim, *args, **kwargs):
        """ 为统一训练方式，几个属性张量必须在子类实现
        input_x,input_y,loss,train_op,predict
        :param data_set: 数据集名称，用来作为模型保存的相对目录
        """
        # for model save & load
        self.name = kwargs['name'] if kwargs.get('name') else self.__class__.__name__  # model name
        self.data_set = data_set
        self.checkpoint_dir = kwargs.get("checkpoint_dir")
        # build model
        self._build(num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim)

    def _build(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.input_def()
        self.embedding_def(num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim)
        self.forward()
        self.predict_def()
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.saver = Saver(max_to_keep=Config.max_to_keep, model_name=self.name,
                           checkpoint_dir=self.checkpoint_dir, relative_dir=self.data_set)

    def build(self):
        raise NotImplemented

    def input_def(self):
        self.input_x = tf.placeholder(tf.int32, [None, Config.sequence_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, Config.num_classes], name="input_y")  # [[1],[1],[-1]]

    def embedding_def(self, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim):
        pass

    def forward(self):
        self.loss = None
        raise NotImplemented

    def predict_def(self):
        pass


class TransXModel(Model):
    def __init__(self, data_set, num_ent_tags, num_rel_tags, ent_emb_dim=Config.ent_emb_dim,
                 rel_emb_dim=Config.rel_emb_dim, *args, **kwargs):
        super().__init__(data_set, num_ent_tags, num_rel_tags, ent_emb_dim, rel_emb_dim, *args, **kwargs)

    def input_def(self):
        sequence_length = Config.sequence_len
        num_classes = Config.num_classes
        batch_size = Config.batch_size
        # [[10630,4,1715],[1422,4,18765]] h,r,t
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")  # [[1],[1],[-1]]
        positive_indices = tf.where(tf.equal(tf.reshape(self.input_y, [-1]), 1))
        negative_indices = tf.where(tf.less(tf.reshape(self.input_y, [-1]), 0.1))
        self.positive_samples = tf.reshape(tf.gather_nd(self.input_x, positive_indices),
                                           [batch_size // 2, sequence_length])
        self.negative_samples = tf.reshape(tf.gather_nd(self.input_x, negative_indices),
                                           [batch_size // 2, sequence_length])
        self.pos_h, self.pos_t, self.pos_r = tf.unstack(self.positive_samples, axis=1)
        self.neg_h, self.neg_t, self.neg_r = tf.unstack(self.negative_samples, axis=1)
        #
        # loss, train_op, predict
        self.predict_h = tf.placeholder(tf.int32, [None], name="predict_h")
        self.predict_t = tf.placeholder(tf.int32, [None], name="predict_t")
        self.predict_r = tf.placeholder(tf.int32, [None], name="predict_r")

    def predict_def(self):
        raise NotImplemented

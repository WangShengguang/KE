import tensorflow as tf

from ke.config import Config
from ke.tf_models.model_utils.saver import Saver


class Model(object):
    """ 轻度封装，实现 model.save 和 model.load
        https://zhuanlan.zhihu.com/p/68899384
        https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
    """

    def __init__(self, data_set, **kwargs):
        """ 为统一训练方式，几个属性张量必须在子类实现
        input_x,input_y,loss,train_op,predict
        :param data_set: 数据集名称，用来作为模型保存的相对目录
        """
        # for model save & load
        self.name = kwargs['name'] if kwargs.get('name') else self.__class__.__name__  # model name
        self.data_set = data_set
        self.checkpoint_dir = kwargs.get("checkpoint_dir")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.saver = None
        self.input_def()

    def input_def(self):
        pass

    def __init_saver(self):
        """ The tf.train.Saver must be created after the variables that you want to restore (or save). Additionally it must be created in the same graph as those variables.
            https://stackoverflow.com/questions/38626435/tensorflow-valueerror-no-variables-to-save-from/38627631
        """
        if self.saver is None:
            relative_dir = self.data_set
            self.saver = Saver(model_name=self.name, checkpoint_dir=self.checkpoint_dir, relative_dir=relative_dir)

    def save(self, sess, loss=0.0, accuracy=0.0):
        """
        :param sess: TensorFlow Session Object
        :type loss: tf.Variable or float
        :type accuracy: float
        """
        self.__init_saver()
        self.saver.save_model(sess, loss=loss, accuracy=accuracy, global_step=self.global_step)
        # logging.info("Model saved in file: {}".format(save_path))

    def load(self, sess, mode="max_step", fail_ok=False):
        """
        :param sess:  tf.Session() Object
        :param mode: max_step, max_acc, min_loss
        :return:
        """
        self.__init_saver()
        self.saver.load_model(sess, mode=mode, fail_ok=fail_ok)
        # logging.info("Model restored from file: {}".format(save_path))


class TransXModel(Model):
    def input_def(self):
        sequence_length = Config.sequence_len
        num_classes = Config.num_classes
        batch_size = Config.batch_size
        # [[10630,4,1715],[1422,4,18765]] h,r,t
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")  # [[1],[1],[-1]]
        # positive_indices = tf.where(tf.equal(tf.reshape(self.input_y, [batch_size, 1]), 1))
        # positive_indices = tf.where(tf.equal(self.input_y, 1))
        # negative_indices = tf.where(tf.equal(self.input_y, 0))
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

import tensorflow as tf

from ke.tf_models.model_utils.saver import Saver


class Model(object):
    """ 轻度封装，实现 model.save 和 model.load
        https://zhuanlan.zhihu.com/p/68899384
        https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
    """

    def __init__(self, **kwargs):
        """ 为统一训练方式，几个属性张量必须在子类实现
        input_x,input_y,loss,train_op,prediction
        """
        self.input_x: tf.placeholder
        self.input_y: tf.placeholder
        self.loss = NotImplemented  # tf.variables
        self.train_op = NotImplemented
        self.prediction = NotImplemented
        # for model save & load
        self.name = kwargs['name'] if kwargs.get('name') else self.__class__.__name__  # model name
        self.checkpoint_dir = kwargs.get("checkpoint_dir")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.saver = None

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

    def __init_saver(self):
        """ The tf.train.Saver must be created after the variables that you want to restore (or save). Additionally it must be created in the same graph as those variables.
            https://stackoverflow.com/questions/38626435/tensorflow-valueerror-no-variables-to-save-from/38627631
        """
        if self.saver is None:
            self.saver = Saver(model_name=self.name, checkpoint_dir=self.checkpoint_dir)

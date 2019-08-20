import glob
import inspect
import logging
import os
import re
import shutil
from pathlib import Path

import tensorflow as tf

from ke.config import Config

session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
)
session_conf.gpu_options.allow_growth = True


class Saver(tf.train.Saver):
    """ https://www.tensorflow.org/guide/saved_model?hl=zh-CN
    主要实现 TensorFlow save 和 load 的统一管理，防止格式不统一造成不方便；
    所有模型使用同一种命名格式保存 model-0.01-0.9.ckpt.meta ...
    """

    def __init__(self, model_name, checkpoint_dir=None, **kwargs):
        """
        :param model_name:  模型名
        :param checkpoint_dir:  模型存储目录
        """
        super().__init__(allow_empty=True, **kwargs)
        self.checkpoint_dir = os.path.join(checkpoint_dir if checkpoint_dir else Config.tf_ckpt_dir, model_name)
        self.ckpt_prefix_template = os.path.join(self.checkpoint_dir, "model-{loss:.3f}-{accuracy:.3f}.ckpt")
        self.meta_path_patten = re.compile("model-(?P<loss>\d+\.\d+)-(?P<acc>[01]\.\d+).ckpt-(?P<step>\d+).meta")

    def save_model(self, sess, global_step, loss=0.0, accuracy=0.0):
        """
        :param sess: TensorFlow Session Object
        :type global_step: tf.Variable or int
        :type loss: tf.Variable or float
        :type accuracy: float
        :return: model_checkpoint_path
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_prefix = self.ckpt_prefix_template.format(loss=loss, accuracy=accuracy)
        model_checkpoint_path = self.save(sess, ckpt_prefix, global_step=global_step)
        # save config.py 将配置参数保存下来
        src_config_path = inspect.getabsfile(Config().__class__)
        dst_config_path = os.path.join(self.checkpoint_dir, Path(src_config_path).name)
        shutil.copyfile(src_config_path, dst_config_path)
        logging.info("* Model save to file: {}".format(model_checkpoint_path))
        return model_checkpoint_path

    def load_model(self, sess, mode=Config.load_model_mode, fail_ok=False):
        """
            https://www.tensorflow.org/guide/saved_model?hl=zh-CN
        :param sess: TensorFlow Session Object
        :param mode: min_loss, max_acc, max_step
        :param fail_ok restore失败是否报错; 默认load失败报错
        :return: load_path
        """
        model_path = self.get_model_path(mode=mode)
        if not Path(model_path + ".meta").is_file():
            logging.info("fail load model from checkpoint_dir : {}... ".format(self.checkpoint_dir))
            if not fail_ok:
                raise ValueError("model_path is not exist, checkpoint_dir: {}".format(self.checkpoint_dir))
        else:
            logging.info("* Model load from file: {}".format(model_path))
            saver = tf.train.import_meta_graph(model_path + ".meta")
            saver.restore(sess, model_path)
        return model_path

    def check_valid(self, model_path):
        """ 确保restore 所需的三个文件都存在
        :param model_path: model_name.meta
        :return: True or False
        """
        path = Path(model_path)
        for suffix in [".meta", ".index", ".data-00000-of-00001"]:
            if not path.with_suffix(suffix).is_file():
                return False
        return True

    def get_model_path(self, mode="min_loss"):
        """
        :param mode: min_loss, max_acc, max_step
        :return: model_path  model_name.meta
                    确保model_name.index,model_name.data-00000-of-00001都存在
        """
        assert mode in ["min_loss", "max_acc", "max_step"], "mode is not exist： {}".format(mode)
        model_paths = glob.glob(os.path.join(self.checkpoint_dir, "*.meta"))
        model_paths = [path for path in model_paths if self.check_valid(path)]
        if model_paths:
            reverse = {"min_loss": True, "max_acc": False, "max_step": False}[mode]  # 从差到好排序
            sorted_model_paths = sorted(
                model_paths,  # loss,acc,global_step
                key=lambda path: float(self.meta_path_patten.search(path).group(mode.split("_")[1])),
                reverse=reverse)
            model_path = sorted_model_paths[-1].strip(".meta")
        else:
            model_path = ""  # 默认返回空路径
        logging.info("\n** get model path:{}\n".format(model_path))
        return model_path


def plot_keras_history(history):
    """
    :param history:  history = model.fit(x,y)
    :return:  None
    """
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.title("accuracy")
    plt.plot(history.history["acc"], color="r", label="train")
    plt.plot(history.history["val_acc"], color="b", label="val")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("loss")
    plt.plot(history.history["loss"], color="r", label="train")
    plt.plot(history.history["val_loss"], color="b", label="val")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()

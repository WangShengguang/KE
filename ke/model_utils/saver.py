import glob
import inspect
import logging
import os
import re
import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework import graph_util

from config import Config


class Saver(tf.train.Saver):
    """ https://www.tensorflow.org/guide/saved_model?hl=zh-CN
    主要实现 TensorFlow save 和 load 的统一管理，防止格式不统一造成不方便；
    所有模型使用同一种命名格式保存 model-0.01-0.9.ckpt.meta ...
    """

    def __init__(self, model_name="", checkpoint_dir=None, relative_dir=None, **kwargs):
        """
        :param model_name:  模型名
        :param checkpoint_dir:  模型存储目录
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        checkpoint_dir = checkpoint_dir if checkpoint_dir else Config.tf_ckpt_dir
        if relative_dir:
            checkpoint_dir = os.path.join(checkpoint_dir, relative_dir)
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        self.ckpt_prefix_template = os.path.join(self.checkpoint_dir, "model-{loss:.3f}-{accuracy:.3f}.ckpt")
        self.meta_path_patten = re.compile("model-(?P<loss>\d+\.\d+)-(?P<acc>[01]\.\d+).ckpt-(?P<step>\d+).meta")
        self.config_saved = False

    def __save_config(self):
        # save config.py 将配置参数保存下来
        if not self.config_saved:
            src_config_path = inspect.getabsfile(Config().__class__)
            dst_config_path = os.path.join(self.checkpoint_dir, Path(src_config_path).with_suffix(".txt").name)
            shutil.copyfile(src_config_path, dst_config_path)
            self.config_saved = True

    def save_model(self, sess, global_step, loss=100.0, accuracy=0.0, **kwargs):
        """
        :param sess: TensorFlow Session Object
        :type global_step: tf.Variable or int
        :type loss: tf.Variable or float
        :type accuracy: float
        :return: model_checkpoint_path
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_prefix = self.ckpt_prefix_template.format(loss=loss, accuracy=accuracy)
        model_checkpoint_path = self.save(sess, ckpt_prefix, global_step=global_step, **kwargs)
        self.__save_config()
        logging.info("* Model save to file: {}".format(model_checkpoint_path))
        return model_checkpoint_path

    def restore_model(self, sess, mode=Config.load_model_mode, fail_ok=False):
        """
            https://www.tensorflow.org/guide/saved_model?hl=zh-CN
        :param sess: TensorFlow Session Object
        :param mode: min_loss, max_acc, max_step
        :param fail_ok restore失败是否报错; 默认load失败报错
        :return: load_path
        """
        model_path = self.get_model_path(mode=mode)
        if not Path(model_path + ".meta").is_file():
            print("fail load model from checkpoint_dir : {}... ".format(self.checkpoint_dir))
            if not fail_ok:
                raise ValueError("model_path is not exist, checkpoint_dir: {}".format(self.checkpoint_dir))
        else:
            _saver = tf.train.import_meta_graph(model_path + ".meta")
            _saver.restore(sess, model_path)
            logging.info("* Model restore from file: {}".format(model_path))
        return model_path

    def check_valid(self, model_path):
        """ 确保restore 所需的三个文件都存在
        :param model_path:
                model_name.meta 包含了网络结构和一些其他信息，所以也包含了上面提到的graph.pb；
                model.checkpoint.data-00000-of-00001保存了模型参数，其他两个文件辅助作用。
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
            is_reverse = True if mode == "min_loss" else False  # 从差到好(loss ↓, acc ↑, step ↑)
            sorted_model_paths = sorted(
                model_paths,  # loss,acc,global_step
                key=lambda path: float(self.meta_path_patten.search(path).group(mode.split("_")[1])),
                reverse=is_reverse)
            model_path = sorted_model_paths[-1].strip(".meta")
        else:
            model_path = ""  # 默认返回空路径
        logging.info("\n** get model path:{}\n".format(model_path))
        return model_path

    def export_graph_pb_from_ckpt(self, model_path=""):
        """ https://blog.csdn.net/guyuealian/article/details/82218092#ckpt-转换成-pb格式
        :type sess: tf.Session
        """
        self.graph_pb_path = os.path.join(self.checkpoint_dir, "pb", self.model_name + ".pb")
        self.graph_output_node_names = ["input_x", "input_y", "prediction"]
        Path(self.graph_pb_path).parent.mkdir(parents=True, exist_ok=True)
        if not model_path:
            model_path = self.get_model_path(mode=Config.load_model_mode)
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_path + ".meta")
            saver.restore(sess, model_path)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                self.graph_output_node_names)
            with tf.gfile.GFile(self.graph_pb_path, "wb") as f:
                f.write(output_graph_def.SerializeToString())

    def load_graph_pb(self):
        if not os.path.isfile(self.graph_pb_path):
            self.export_graph_pb_from_ckpt()
        with tf.gfile.GFile(self.graph_pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=self.model_name)
        return graph


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

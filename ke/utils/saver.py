import inspect
import logging
import os
import re
import shutil
from pathlib import Path

import tensorflow as tf

from config import CkptConfig


class Saver(object):
    """ https://www.tensorflow.org/guide/saved_model?hl=zh-CN
    主要实现 TensorFlow save 和 load 的统一管理，防止格式不统一造成不方便；
    所有模型使用同一种命名格式保存 model-0.01-0.9.ckpt.meta ...
    """

    def __init__(self, data_set, model_name, to_save=True):
        """
        :param model_name:  模型名
        :param checkpoint_dir:  模型存储目录
        """
        self.config = CkptConfig(data_set, model_name)
        self.checkpoint_dir = self.config.tf_ckpt_dir
        self.ckpt_prefix_template = "model-{loss:.3f}-{accuracy:.3f}.ckpt"
        self.meta_path_patten = re.compile(
            "model-(?P<min_loss>\d+\.\d+)-(?P<max_acc>[01]\.\d+).ckpt-(?P<max_step>\d+).meta")
        self.modes = ["max_acc", "min_loss", "max_step"]
        if to_save:
            self.savers = {key: tf.train.Saver(max_to_keep=self.config.max_to_keep) for key in self.modes}
        self.config_saved = False

    def __save_config_file(self):
        # save config.py 将配置参数保存下来
        if not self.config_saved:
            src_config_path = inspect.getabsfile(self.config.__class__)
            dst_config_path = os.path.join(self.checkpoint_dir, Path(src_config_path).with_suffix(".txt").name)
            shutil.copyfile(src_config_path, dst_config_path)
            self.config_saved = True

    def save_model(self, sess, global_step, mode="max_step", loss=11111.0, accuracy=0.0):
        """
        :param sess: TensorFlow Session Object
        :param mode: min_loss, max_acc, max_step
        :type global_step: tf.Variable or int
        :type loss: tf.Variable or float
        :type accuracy: float
        :return: model_checkpoint_path
        """
        assert mode in self.modes, "mode is not exist： {}".format(mode)
        self.__save_config_file()
        _ckpt_dir = os.path.join(self.checkpoint_dir, mode)
        os.makedirs(_ckpt_dir, exist_ok=True)
        ckpt_prefix = os.path.join(_ckpt_dir, self.ckpt_prefix_template.format(loss=loss, accuracy=accuracy))
        saver = self.savers[mode]
        ckpt_save_path = saver.save(sess, ckpt_prefix, global_step=global_step)
        logging.info("* Model save to file: {}".format(ckpt_save_path))
        return ckpt_save_path

    def restore_model(self, sess, mode=CkptConfig.load_model_mode, fail_ok=False):
        """
            https://www.tensorflow.org/guide/saved_model?hl=zh-CN
        :param sess: TensorFlow Session Object
        :param mode: min_loss, max_acc, max_step
        :param fail_ok restore失败是否报错; 默认load失败报错
        :return: load_path
        """
        assert mode in self.modes, "mode is not exist： {}".format(mode)
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

    def get_model_path(self, mode="max_step"):
        """
        :param mode: min_loss, max_acc, max_step
        :return: model_path  model_name.meta
                    确保model_name.index,model_name.data-00000-of-00001都存在
        """
        assert mode in self.modes, "mode is not exist： {}".format(mode)
        ckpt_dir = os.path.join(self.checkpoint_dir, mode)
        model_paths = Path(ckpt_dir).glob("*.meta")
        model_paths = [str(path) for path in model_paths if self.check_valid(path)]
        print("* find {} models from {}".format(len(model_paths), ckpt_dir))
        if model_paths:
            is_reverse = True if mode == "min_loss" else False  # 从差到好(loss ↓, acc ↑, step ↑)
            sorted_model_paths = sorted(
                model_paths,  # loss,acc,global_step
                key=lambda _path: float(self.meta_path_patten.search(_path).group(mode)),
                reverse=is_reverse)
            model_path = sorted_model_paths[-1].strip(".meta")
        else:
            model_path = ""  # 默认返回空路径
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

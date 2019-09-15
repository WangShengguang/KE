import inspect
import logging
import os
import re
import shutil
from pathlib import Path

import tensorflow as tf

from config import Config


class Saver(tf.train.Saver):
    """ https://www.tensorflow.org/guide/saved_model?hl=zh-CN
    主要实现 TensorFlow save 和 load 的统一管理，防止格式不统一造成不方便；
    所有模型使用同一种命名格式保存 model-0.01-0.9.ckpt.meta ...
    """

    def __init__(self, model_name="", checkpoint_dir=Config.tf_ckpt_dir, relative_dir=None, **kwargs):
        """
        :param model_name:  模型名
        :param checkpoint_dir:  模型存储目录
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.base_checkpoint_dir = os.path.join(checkpoint_dir, relative_dir) if relative_dir else checkpoint_dir
        self.config_saved = False
        self.__init_model_path()

    def __init_model_path(self):
        self.checkpoint_dir = os.path.join(self.base_checkpoint_dir, self.model_name)
        self.min_loss_ckpt_dir = os.path.join(self.checkpoint_dir, "min_loss")
        self.max_acc_ckpt_dir = os.path.join(self.checkpoint_dir, "max_acc")
        self.max_step_ckpt_dir = os.path.join(self.checkpoint_dir, "max_step")
        for _dir in [self.max_acc_ckpt_dir, self.min_loss_ckpt_dir, self.max_step_ckpt_dir]:
            os.makedirs(_dir, exist_ok=True)
        self.ckpt_prefix_template = "model-{loss:.3f}-{accuracy:.3f}.ckpt"
        self.meta_path_patten = re.compile(
            "model-(?P<min_loss>\d+\.\d+)-(?P<max_acc>[01]\.\d+).ckpt-(?P<max_step>\d+).meta")
        self.min_loss_ckpt_path_tmpl = os.path.join(self.min_loss_ckpt_dir, self.ckpt_prefix_template)
        self.max_acc_ckpt_path_tmpl = os.path.join(self.max_acc_ckpt_dir, self.ckpt_prefix_template)
        self.max_step_ckpt_path_tmpl = os.path.join(self.max_step_ckpt_dir, self.ckpt_prefix_template)
        self.ckpt_tmpls = {"min_loss": self.min_loss_ckpt_path_tmpl,
                           "max_acc": self.max_acc_ckpt_path_tmpl,
                           "max_step": self.max_step_ckpt_path_tmpl}

    def __save_config(self):
        # save config.py 将配置参数保存下来
        if not self.config_saved:
            src_config_path = inspect.getabsfile(Config().__class__)
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
        self.__save_config()
        ckpt_prefix = self.ckpt_tmpls[mode].format(loss=loss, accuracy=accuracy)
        model_checkpoint_path = self.save(sess, ckpt_prefix, global_step=global_step, )
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

    def get_model_path(self, mode="max_step"):
        """
        :param mode: min_loss, max_acc, max_step
        :return: model_path  model_name.meta
                    确保model_name.index,model_name.data-00000-of-00001都存在
        """
        assert mode in ["min_loss", "max_acc", "max_step"], "mode is not exist： {}".format(mode)
        model_paths = Path(self.checkpoint_dir).joinpath(mode).rglob("*.meta")
        model_paths = [path for path in model_paths if self.check_valid(path)]
        print("model_paths: {}".format(model_paths))
        if model_paths:
            is_reverse = True if mode == "min_loss" else False  # 从差到好(loss ↓, acc ↑, step ↑)
            sorted_model_paths = sorted(
                model_paths,  # loss,acc,global_step
                key=lambda path: float(self.meta_path_patten.search(str(path)).group(mode)),
                reverse=is_reverse)
            model_path = str(sorted_model_paths[-1]).strip(".meta")
        else:
            model_path = ""  # 默认返回空路径
        logging.info("\n** get model path:{}\n".format(model_path))
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

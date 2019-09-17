import os
import random

import numpy as np
import tensorflow as tf

__all__ = ["Config"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir
data_dir = os.path.join(root_dir, "benchmarks")

# random seed
rand_seed = 1234
random.seed(rand_seed)
np.random.seed(rand_seed)


class DataConfig(object):
    """
        数据和模型所在文件夹
    """
    model_ckpt_dir = os.path.join(root_dir, "model_ckpt")
    tf_ckpt_dir = os.path.join(model_ckpt_dir, "tf")
    keras_ckpt_dir = os.path.join(model_ckpt_dir, "keras")


class TrainConfig(DataConfig):
    sequence_len = 3  # (h,r,t)
    num_classes = 1  # 0 or 1
    # neg_label = -1.0  # 负样本标签
    batch_size = 8
    epoch_nums = 1000
    # margin loss
    learning_rate = 0.0001
    l2_reg_lambda = 0.001
    dropout_keep_prob = 0.8
    margin = 1.0
    #
    ent_emb_dim = 128
    rel_emb_dim = 128
    hidden_dim = 128
    # early stop
    max_epoch_nums = 100
    min_epoch_nums = 5

    # lawdata 10000
    patience = 0.0001
    patience_num = 5
    # model save & load
    load_pretrain = True  # 断点续训
    max_to_keep = 2
    save_step = 1000
    #
    random_seed = 1234
    np.random.seed(random_seed)


class Evaluate(TrainConfig):
    load_model_mode = "max_step"
    # load_model_mode = "min_loss"
    # load_model_mode = "max_acc"  # mrr


class TfConfig(object):
    """
        TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
        TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
        TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
        TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    """
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1,
                                                            # 随着进程逐渐增加显存占用，而不是一下占满
                                                            allow_growth=True))


class Config(Evaluate, TfConfig):
    train_count = -1  # all count
    valid_count = 1000
    test_count = 100
    # np.random.seed(1234)
    # tf.set_random_seed(1234)
    # torch.seed(1234)
    # random.seed(1234)

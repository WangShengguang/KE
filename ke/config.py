import os

__all__ = ["Config"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)


class DataConfig(object):
    """
        数据和模型所在文件夹
    """
    data_dir = os.path.join(root_dir, "benchmarks")
    model_ckpt_dir = os.path.join(root_dir, "model_ckpt")
    tf_ckpt_dir = os.path.join(model_ckpt_dir, "tf")
    keras_ckpt_dir = os.path.join(model_ckpt_dir, "keras")


class TrainConfig(DataConfig):
    sequence_len = 3  # (h,r,t)
    num_classes = 1
    batch_size = 128
    epoch_nums = 100
    # early stop
    max_epoch_nums = 20
    min_epoch_nums = 5
    patience = 0.02
    patience_num = 3
    # model save & load
    load_pretrain = False  # 断点续训
    max_to_keep = 10
    save_step = 200


class Evaluate(TrainConfig):
    load_model_mode = "max_step"


class Config(Evaluate):
    pass

import os
import re

import numpy as np

from ke.config import data_dir, Config
from sklearn.utils import shuffle


class DataHelper(object):
    def __init__(self, data_set):
        """
        :param data_set: 数据集名称
                在benchmarks目录下，需包含一下几个文件; 第一行为文件行数-1 = 样本数
                train2id.txt, valid.txt, test.txt :  h,t,r
                entity2id.txt, relation2id.txt
        """
        self.data_set = data_set  # 数据集名称
        self.data = {}  # {"train": [(h,t,r), ...], "valid":  [(h,t,r), ...], "test":  [(h,t,r), ...]}
        self.entity2id = {}
        self.relation2id = {}
        self.load_data()

    def load_data(self):
        """ 原始文件读入内存 self.data, self.entity2id, self.entity2id
            count, *lines = f.readlines()
        """

        def read_data(file_name):
            file_path = os.path.join(data_dir, self.data_set, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                count = int(f.readline())
                lines = [re.sub("\s+", " ", line).strip() for line in f if line.strip()]
            assert count == len(lines), file_path
            return lines

        self.data = {}
        count_limit = {"train": Config.train_count, "valid": Config.valid_count, "test": Config.test_count}  # 样本数限制
        for data_type in ["train", "valid", "test"]:
            lines = read_data(f"{data_type}2id.txt")
            self.data[data_type] = [(int(h), int(t), int(r)) for line in lines for h, t, r in [line.split(" ")]][
                                   :count_limit[data_type]]
        lines = read_data("entity2id.txt")
        self.entity2id = {entity: int(id) for line in lines for entity, id in [line.split(" ")]}
        lines = read_data("relation2id.txt")
        self.relation2id = {relation: int(id) for line in lines for relation, id in [line.split(" ")]}

    def get_samples(self, data_type):
        """
        :param data_type:  train,valid,test
        """
        positive_samples = self.data[data_type]
        pos_samples_set = set(positive_samples)
        entity_ids = list(self.entity2id.values())
        negative_samples = []
        for h, t, r in positive_samples:
            while (h, t, r) in pos_samples_set:
                e = np.random.choice(entity_ids)
                if np.random.randint(0, 2):  # [0,1]
                    h = e
                else:
                    t = e
            negative_samples.append((h, t, r))
        assert len(positive_samples) == len(negative_samples)
        return positive_samples, negative_samples

    def mix_batch_iter(self, positive_samples, negative_samples, batch_size, _shuffle=True, neg_label=-1.0):
        data_size = len(positive_samples)
        order = list(range(data_size))
        if _shuffle:
            np.random.shuffle(order)
        semi_batch_size = (batch_size // 2)
        for batch_step in range(data_size // semi_batch_size):
            batch_idxs = order[batch_step * semi_batch_size:(batch_step + 1) * semi_batch_size]
            if len(batch_idxs) != semi_batch_size:
                continue
            _positive_samples = [positive_samples[idx] for idx in batch_idxs]
            _negative_samples = [negative_samples[idx] for idx in batch_idxs]
            x_batch = _positive_samples + _negative_samples
            y_batch = [[1.0]] * semi_batch_size + [[neg_label]] * semi_batch_size
            yield np.asarray(x_batch), np.asarray(y_batch)

    def concat_batch_iter(self, positive_samples, negative_samples, batch_size, _shuffle=True, neg_label=-1.0):
        if _shuffle:
            positive_samples, negative_samples = shuffle(positive_samples, negative_samples)
        x_data = positive_samples + negative_samples
        y_data = [[1.0]] * len(positive_samples) + [[neg_label]] * len(negative_samples)
        data_size = len(x_data)
        order = list(range(data_size))
        for batch_step in range(data_size // batch_size):
            batch_idxs = order[batch_step * batch_size:(batch_step + 1) * batch_size]
            if len(batch_idxs) != batch_size:
                continue
            x_batch = [x_data[idx] for idx in batch_idxs]
            y_batch = [y_data[idx] for idx in batch_idxs]
            yield np.asarray(x_batch), np.asarray(y_batch)

    def batch_iter(self, positive_samples, negative_samples, batch_size, mode, _shuffle=True, neg_label=-1.0):
        """
        :param positive_samples:
        :param negative_samples:
        :param batch_size:
        :param mode:  mix,concat
                     mix: x_batch = [positive+neggtive]
                     concat :x_batch = [positive] + [negtive]
        :param _shuffle:
        :param neg_label:
        :return:
        """
        if mode == "mix":
            for x_batch, y_batch in self.mix_batch_iter(positive_samples, negative_samples, batch_size,
                                                        _shuffle=True, neg_label=-1.0):
                yield x_batch, y_batch
        elif mode == "concat":
            for x_batch, y_batch in self.concat_batch_iter(positive_samples, negative_samples, batch_size,
                                                           _shuffle=True, neg_label=-1.0):
                yield x_batch, y_batch
        else:
            raise ValueError(mode)

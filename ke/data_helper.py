import os
import re

import numpy as np

from ke.config import data_dir


class DataHelper(object):
    def __init__(self, data_set):
        """
        :param data_set: 数据集名称
                在benchmarks目录下，需包含一下几个文件; 第一行为文件行数-1 = 样本数
                train2id.txt, valid.txt, test.txt :  h,t,r
                entity2id.txt, relation2id.txt
        """
        self.data_set = data_set  # 数据集名称
        self.data = {}  # {"train": [(e,r,t), ...], "valid":  [(e,r,t), ...], "test":  [(e,r,t), ...]}
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
        for data_type in ["train", "valid", "test"]:
            lines = read_data(f"{data_type}2id.txt")
            self.data[data_type] = [(int(h), int(t), int(r)) for line in lines for h, t, r in [line.split(" ")]]
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
            while (h, t, r) in pos_samples_set:  # TODO replace r，not only h,t
                e = np.random.choice(entity_ids)
                if np.random.choice([True, False]):
                    h = e
                else:
                    t = e
            negative_samples.append((h, t, r))
        assert len(positive_samples) == len(negative_samples)
        return positive_samples, negative_samples

    def batch_iter(self, data_type, batch_size, _shuffle=True, mode="hrt", neg_label=-1):
        positive_samples, negative_samples = self.get_samples(data_type)
        data_size = len(positive_samples)
        order = list(range(data_size))
        if _shuffle:
            np.random.shuffle(order)
        _batch_size = (batch_size // 2)
        for batch_step in range(data_size // _batch_size):
            # fetch sentences and tags
            batch_idxs = order[batch_step * _batch_size:(batch_step + 1) * _batch_size]
            _positive_samples = [positive_samples[idx] for idx in batch_idxs]
            _negative_samples = [negative_samples[idx] for idx in batch_idxs]
            x_batch, y_batch = [], []
            for (h, t, r) in _positive_samples:
                x = (h, t, r) if mode == "htr" else (h, r, t)  # 交换了位置
                x_batch.append(x)
                y_batch.append([1])
            for (h, t, r) in _negative_samples:
                x = (h, t, r) if mode == "htr" else (h, r, t)  # 交换了位置
                x_batch.append(x)  # 交换了位置
                y_batch.append([neg_label])
            yield np.asarray(x_batch), np.asarray(y_batch)

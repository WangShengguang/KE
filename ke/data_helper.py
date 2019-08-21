import os
import re

import numpy as np
from sklearn.utils import shuffle

from ke.config import Config

np.random.seed(1234)
data_dir = Config.data_dir


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
            while (h, t, r) in pos_samples_set:
                e = np.random.choice(entity_ids)
                if np.random.choice([True, False]):
                    h = e
                else:
                    t = e
            negative_samples.append((h, t, r))
        return positive_samples, negative_samples

    def batch_iter(self, data_type, batch_size, epoch_nums, _shuffle=True):
        positive_samples, negative_samples = self.get_samples(data_type)
        x_data = positive_samples + negative_samples
        y_data = [[1]] * len(positive_samples) + [[0]] * len(negative_samples)  # y为一维向量
        for epoch in range(epoch_nums):
            if _shuffle:
                x_data, y_data = shuffle(x_data, y_data, random_state=epoch)
            x_batch = []
            y_batch = []
            for x_sample, y_sample in zip(x_data, y_data):
                x_batch.append(x_sample)  # 注意，交换了位置
                y_batch.append(y_sample)
                if len(x_batch) == batch_size:
                    yield np.asarray(x_batch), np.asarray(y_batch)
                    x_batch = []
                    y_batch = []

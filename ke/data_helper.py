import os
import re

import numpy as np
from sklearn.utils import shuffle

from ke.config import Config

data_dir = Config.data_dir


class DataHelper(object):
    def __init__(self, data_set):
        """
        :param data_set: 数据集名称
                在benchmarks目录下，需包含一下几个文件; 第一行为文件行数-1 = 样本数
                train2id.txt, valid.txt, test.txt :  e,r,t
                entity2id.txt, relation2id.txt
        """
        self.data_set = data_set  # 数据集名称
        self.data = {}  # {"train": [(e,r,t), ...], "valid":  [(e,r,t), ...], "test":  [(e,r,t), ...]}
        self.entity2id = {}
        self.relation2id = {}
        self.init()

    def init(self):
        """ 原始文件读入内存 self.data, self.entity2id, self.entity2id
        """
        file_path_template = os.path.join(data_dir, self.data_set, "{}2id.txt")
        self.data = {"train": [], "valid": [], "test": []}
        s_patten = re.compile("\s+")
        for data_type in ["train", "valid", "test"]:
            with open(file_path_template.format(data_type), "r", encoding="utf-8") as f:
                count, *lines = f.readlines()
                lines = [s_patten.sub(" ", line).strip() for line in lines if line.strip()]
                assert int(count.strip()) == len(
                    lines), f"{self.data_set}-{data_type} , count: {count},len(lines): {len(lines)}"
                for line in lines:
                    e1, e2, r = line.split(" ")
                    self.data[data_type].append((int(e1), int(e2), int(r)))
        with open(os.path.join(data_dir, self.data_set, "entity2id.txt"), "r", encoding="utf-8") as f:
            count, *lines = f.readlines()
            lines = [s_patten.sub(" ", line).strip() for line in lines if line.strip()]
        self.entity2id = {entity: int(id) for line in lines for entity, id in [line.split(" ")]}
        with open(os.path.join(data_dir, self.data_set, "relation2id.txt"), "r", encoding="utf-8") as f:
            count, *lines = f.readlines()
            lines = [s_patten.sub(" ", line).strip() for line in lines if line.strip()]
        self.relation2id = {relation: int(id) for line in lines for relation, id in [line.split(" ")]}
        print(" * data helper init done ...")

    def get_samples(self, data_type, expand_negetive=True):
        """
        :param data_type:  train,valid,test
        :param expand_negetive bool
        :return:
        """
        x_data = self.data[data_type]
        y_data = [[1]] * len(x_data)  # y为一维向量
        # if data_type == "train":
        if expand_negetive:  # generate negetive samples
            pos_samples_set = set(x_data)
            entity_ids = list(self.entity2id.values())
            for e1, e2, r in pos_samples_set:  # 使用x_data造成x_data增长，迭代无法结束
                while (e1, e2, r) in pos_samples_set:
                    e2 = np.random.choice(entity_ids)
                x_data.append((e1, e2, r))
                y_data.append([0])  # y为一维向量
        return x_data, y_data

    def batch_iter(self, data_type, batch_size, epoch_nums, _shuffle=True):
        x_data, y_data = self.get_samples(data_type)
        for epoch in range(epoch_nums):
            if _shuffle:
                x_data, y_data = shuffle(x_data, y_data, random_state=epoch)
            x_batch = []
            y_batch = []
            for (h, t, r), y_sample in zip(x_data, y_data):
                x_batch.append((h, t, r))  # 注意，交换了位置
                y_batch.append(y_sample)
                if len(x_batch) == batch_size:
                    yield np.asarray(x_batch), np.asarray(y_batch)
                    x_batch = []
                    y_batch = []

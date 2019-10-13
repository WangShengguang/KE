import os
import random
import re

import numpy as np

from config import data_dir, Config


class DataHelper(object):
    def __init__(self, data_set, model_name):
        """
        :param data_set: 数据集名称
                在benchmarks目录下，需包含一下几个文件; 第一行为文件行数-1 = 样本数
                train2id.txt, valid.txt, test.txt :  h,t,r
                entity2id.txt, relation2id.txt
        :param model_name 这两个模型使用同一个embedding 矩阵，所以需要统一编码entity2id和relation2id，不能分开编码
        """
        self.data_set = data_set  # 数据集名称
        self.model_name = model_name
        self.data = {}  # {"train": [(h,t,r), ...], "valid":  [(h,t,r), ...], "test":  [(h,t,r), ...]}
        self.entity2id = {}
        self.relation2id = {}
        self.load_data()
        #
        self.inited_negative = False

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
        # count_limit = {"train": Config.train_count, "valid": Config.valid_count, "test": Config.test_count}  # 样本数限制
        for data_type in ["train", "valid", "test"]:
            lines = read_data(f"{data_type}2id.txt")
            self.data[data_type] = [(int(h), int(t), int(r)) for line in lines for h, t, r in [line.split(" ")]]
        lines = read_data("entity2id.txt")
        self.entity2id = {entity: int(id) for line in lines for entity, id in [line.split(" ")]}
        # 这两个模型使用同一个embedding 矩阵，所以需要统一编码，不能分开编码
        # rel_start_id = len(self.entity2id) if self.model_name in ["ConvKB", "TransformerKB"] else 0
        lines = read_data("relation2id.txt")
        self.relation2id = {relation: int(_id) for line in lines for relation, _id in [line.split(" ")]}

    def init_negative_samples(self, positive_samples):
        if not self.inited_negative:
            self.pos_samples_set = set(positive_samples)
            self.entity_ids = list(self.entity2id.values())
            self.neg_batch_count = 0
            self.inited_negative = True

    def get_batch_negative_samples(self, batch_positive_samples):
        """
        :param data_type:  train,valid,test
        """
        batch_negative_samples = []
        # state = self.neg_batch_count % 2  # [0,1] ,每个批次只随机替换 head or tail
        # self.neg_batch_count += 1
        for (h, t, r) in batch_positive_samples:
            while (h, t, r) in self.pos_samples_set:
                e = np.random.choice(self.entity_ids)
                if random.randint(0, 1):
                    h = e
                else:
                    t = e
            batch_negative_samples.append((h, t, r))
        assert len(batch_positive_samples) == len(batch_negative_samples)
        return batch_negative_samples

    def batch_iter(self, data_type, batch_size, _shuffle=True):
        positive_samples = self.data[data_type]
        self.init_negative_samples(positive_samples)
        semi_data_size = len(positive_samples)
        order = list(range(semi_data_size))
        if _shuffle:
            random.shuffle(order)
        semi_batch_size = batch_size // 2
        for batch_step in range(semi_data_size // semi_batch_size):
            # print("batch_step： {}".format(batch_step))
            batch_idxs = order[batch_step * semi_batch_size:(batch_step + 1) * semi_batch_size]
            if len(batch_idxs) != semi_batch_size:
                continue
            _positive_samples = [positive_samples[idx] for idx in batch_idxs]
            _negative_samples = self.get_batch_negative_samples(_positive_samples)
            x_batch = _positive_samples + _negative_samples
            y_batch = [[1.0]] * semi_batch_size + [[-1.0]] * semi_batch_size
            yield np.asarray(x_batch), np.asarray(y_batch)

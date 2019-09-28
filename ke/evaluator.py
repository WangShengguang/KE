import logging
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from config import Config
from ke.data_helper import DataHelper
from ke.utils.saver import Saver


class Predictor(object):
    def __init__(self, model_name, data_set, sess=None):
        self.model_name = model_name
        self.data_set = data_set
        # self.rank_metrics = RankMetrics()
        self.data_helper = DataHelper(data_set=data_set, model_name=model_name)
        self.entity_nums = len(self.data_helper.entity2id)
        self.relation_nums = len(self.data_helper.relation2id)
        self.load_model()

    def load_model(self):
        graph = tf.Graph()
        self.sess = tf.Session(config=Config.session_conf, graph=graph)
        with graph.as_default(), self.sess.as_default():  # self 无法load TransformerKB
            model_path = Saver(self.data_set, self.model_name, to_save=False).restore_model(self.sess)
            print("load model from : {}".format(model_path))
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
            self.prediction = graph.get_operation_by_name("predict").outputs[0]
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

    def predict(self, batch_h: List, batch_t: List, batch_r: List):
        """ 用于预测
        :param batch_h: [0,6,3,...,h_id]
        :param batch_t: [0,6,3,...,t_id]
        :param batch_r: [0,6,3,...,r_id]
        :return:
        """
        input_size, _batch_size = len(batch_h), Config.batch_size
        x_data = list(zip(batch_h, batch_t, batch_r)) + [(0, 0, 0)] * (_batch_size - input_size % _batch_size)
        prediction = []
        for input_x in [x_data[i:i + _batch_size] for i in range(0, len(x_data), _batch_size)]:
            _pred = self.sess.run(self.prediction, feed_dict={self.input_x: input_x, self.dropout_keep_prob: 1.0})
            prediction.extend(_pred.flatten().tolist())
        return np.array(prediction[:input_size])

    # link Predict 链接预测，预测头实体or尾实体
    def predict_head_entity(self, t, r):
        r'''This mothod predicts the top k head entities given tail entity and relation.

        Args:
            t (int): tail entity id
            r (int): relation id
            k (int): top k head entities
        Returns:
            list: k possible head entity ids
        '''
        test_h = list(range(self.entity_nums))
        test_r = [r] * self.entity_nums
        test_t = [t] * self.entity_nums
        predictions = self.predict(test_h, test_t, test_r)
        head_ids = predictions.flatten().argsort().tolist()  # dist越小越相似
        # print(head_ids)
        return head_ids

    def predict_tail_entity(self, h, r):
        r'''This mothod predicts the top k tail entities given head entity and relation.

        Args:
            h (int): head entity id
            r (int): relation id
            k (int): top k tail entities
        Returns:
            list: k possible tail entity ids
        '''
        test_h = [h] * self.entity_nums
        test_r = [r] * self.entity_nums
        test_t = list(range(self.entity_nums))
        predictions = self.predict(test_h, test_t, test_r)
        tail_ids = predictions.flatten().argsort().tolist()  # dist越小越相似
        # print(tail_ids)
        return tail_ids

    def predict_relation(self, h, t):
        r'''This methods predict the relation id given head entity and tail entity.

        Args:
            h (int): head entity id
            t (int): tail entity id
            k (int): top k relations

        Returns:
            list: k possible relation ids
        '''
        test_h = [h] * self.relation_nums
        test_r = list(range(self.relation_nums))
        test_t = [t] * self.relation_nums
        predictions = self.predict(test_h, test_t, test_r)
        relations = predictions.flatten().argsort().tolist()  # dist越小越相似
        # print(relations)
        return relations

    def predict_triple(self, h, t, r, thresh=None):
        r'''This method tells you whether the given triple (h, t, r) is correct of wrong

        Args:
            h (int): head entity id
            t (int): tail entity id
            r (int): relation id
            thresh (fload): threshold for the triple
        '''
        prediction = self.predict([h], [t], [r])
        return prediction


class Evaluator(Predictor):
    def __init__(self, model_name, data_set, data_type="test", sess=None):
        super().__init__(model_name, data_set, sess)
        self.data_type = data_type

    def get_valid_lr(self):
        valid_count = len(self.data_helper.data[self.data_type])
        valid_left, valid_right = {}, {}
        valid_samples = self.data_helper.data[self.data_type]  # [(h,t,r)]
        for i in range(valid_count):
            if (valid_samples[i][-1] != valid_samples[i - 1][-1]):
                valid_right[valid_samples[i - 1][-1]] = i - 1
                valid_left[valid_samples[i][-1]] = i
        valid_left[valid_samples[0][-1]] = 0
        valid_right[valid_samples[valid_count - 1][-1]] = valid_count - 1
        return valid_left, valid_right

    def get_best_threshold(self, positives_score, negatives_score):
        interval = 0.01
        bestAcc, max_score, min_score = 0.0, 0.0, 1000
        bestThresh = 0.0
        rel_threshold = {}
        valid_left, valid_right = self.get_valid_lr()
        for r in self.data_helper.relation2id.values():
            if valid_left[r] == -1:
                continue
            total = (valid_right[r] - valid_left[r] + 1) * 2

            min_score = min(min_score, negatives_score[valid_left[r]])
            max_score = max(max_score, positives_score[valid_left[r]])
            for i in range(valid_left[r] + 1, valid_right[r] + 1):
                min_score = min(min_score, positives_score[i], negatives_score[i])
                max_score = max(max_score, positives_score[i], negatives_score[i])

            n_interval = int((max_score - min_score) / interval)
            for i in range(n_interval + 1):
                tmpThresh = min_score + i * interval
                correct = 0
                for j in range(valid_left[r], valid_right[r] + 1):
                    if (positives_score[j] <= tmpThresh):
                        correct += 1
                    if (negatives_score[j] > tmpThresh):
                        correct += 1

                tmpAcc = 1.0 * correct / total
                if i == 0:
                    bestThresh = tmpThresh
                    bestAcc = tmpAcc
                elif tmpAcc > bestAcc:
                    bestAcc = tmpAcc
                    bestThresh = tmpThresh

            rel_threshold[r] = bestThresh
        return rel_threshold

    def test(self):
        ranks = []
        ranks_left = []
        ranks_right = []
        hits = {1: [], 3: [], 10: []}
        hits_left = {1: [], 3: [], 10: []}
        hits_right = {1: [], 3: [], 10: []}
        total = len(self.data_helper.data[self.data_type])
        logging.info("*model:{} {}, test start, {}: {} ".format(self.model_name, self.data_set, self.data_type, total))
        for step, (h, t, r) in enumerate(tqdm(self.data_helper.data[self.data_type],
                                              desc="{} Evaluator test".format(self.model_name))):
            pred_head_ids = self.predict_head_entity(t, r)
            rank_left = pred_head_ids.index(h) + 1
            pred_tail_ids = self.predict_tail_entity(h, r)
            rank_right = pred_tail_ids.index(t) + 1
            ranks.append(rank_left), ranks.append(rank_right)
            ranks_left.append(rank_left)
            ranks_right.append(rank_right)

            if rank_left <= 1:
                hits[1].append(1)
                hits_left[1].append(1)
            elif rank_left <= 3:
                hits[3].append(1)
                hits_left[3].append(1)
            elif rank_left <= 10:
                hits[10].append(1)
                hits_left[10].append(1)

            if rank_right <= 1:
                hits[1].append(1)
                hits_right[1].append(1)
            elif rank_right <= 3:
                hits[3].append(1)
                hits_right[1].append(1)
            elif rank_right <= 10:
                hits[10].append(1)
                hits_right[1].append(1)
            logging.info("*model:{} {}, test step: {}/{}".format(self.model_name, self.data_set, step, total))
        for k in [1, 3, 10]:
            hits[k] = np.mean(hits[k])
        # MR
        mr = np.mean(ranks)
        mr_left = np.mean(ranks_left)
        mr_right = np.mean(ranks_right)
        # MRR
        mrr = np.mean(1. / np.array(ranks))
        mrr_left = np.mean(1. / np.array(ranks_left))
        mrr_right = np.mean(1. / np.array(ranks_right))
        metrics = {
            "left": {"MR": mr_left, "MRR": mrr_left,
                     "Hit@10": hits_left[10], "Hit@3": hits_left[3], "Hit@1": hits_left[1]},
            "right": {"MR": mr_right, "MRR": mrr_right,
                      "Hit@10": hits_right[10], "Hit@3": hits_right[3], "Hit@1": hits_right[1]},
            "ave": {"MR": mr, "MRR": mrr,
                    "Hit@10": hits[10], "Hit@3": hits[3], "Hit@1": hits[1]}
        }
        return metrics

    def test_link_prediction(self):
        """
        链接预测，预测头实体或尾实体
        """
        metrics_li = []
        total = len(self.data_helper.data[self.data_type])
        logging.info("* model:{},{} test_link_prediction start, {}: {} ".format(self.model_name,
                                                                                self.data_set, self.data_type, total))
        for step, (h, t, r) in enumerate(tqdm(self.data_helper.data[self.data_type],
                                              desc="{} {} test_link_prediction".format(self.model_name,
                                                                                       self.data_set))):
            pred_head_ids = self.predict_head_entity(t, r)
            _metrics = get_rank_hit_metrics(y_id=h, pred_ids=pred_head_ids)
            metrics_li.append(_metrics)
            pred_tail_ids = self.predict_tail_entity(h, r)
            _metrics = get_rank_hit_metrics(y_id=t, pred_ids=pred_tail_ids)
            metrics_li.append(_metrics)
            logging.info("*model:{} {}, test_link_prediction step: {}/{}".format(self.model_name,
                                                                                 self.data_set, step, total))
        metrics = {}
        for metric_name in metrics_li[0].keys():
            metrics[metric_name] = sum([_metrics[metric_name] for _metrics in metrics_li]) / len(metrics_li)
        mr, mrr, hit_10, hit_3, hit_1 = (metrics["MR"], metrics["MRR"],
                                         metrics["HITS@10"], metrics["HITS@3"], metrics["HITS@1"])
        # logging.info("mr:{:.4f}, mrr:{:.4f}, hit_1:{:.4f}, hit_3:{:.4f}, hit_10:{:.4f}".format(
        #     mr, mrr, hit_1, hit_3, hit_10))
        return mr, mrr, hit_10, hit_3, hit_1

    def test_triple_classification(self):
        y_true = []
        y_pred = []
        positive_samples, negative_samples = [], []
        # todo get positive_samples, negative_samples
        _positive_samples, _negative_samples = np.asarray(positive_samples), np.asarray(negative_samples)
        positive_score = self.predict(batch_h=_positive_samples[:, 0], batch_t=_positive_samples[:, 1],
                                      batch_r=_positive_samples[:, 2])
        negative_score = self.predict(batch_h=_negative_samples[:, 0], batch_t=_negative_samples[:, 1],
                                      batch_r=_negative_samples[:, 2])
        rel_threshold = self.get_best_threshold(positive_score, negative_score)
        total = len(positive_samples) + len(negative_samples)
        logging.info("* model:{} {}, test_triple_classification start, {}: {} ".format(
            self.model_name, self.data_set, self.data_type, total))
        for x_batch, y_batch in tqdm(
                self.data_helper.batch_iter(data_type=self.data_type, batch_size=Config.batch_size),
                total=total / Config.batch_size,
                desc="{} {} test_triple_classification".format(self.model_name, self.data_set)):
            prediction = self.predict(batch_h=x_batch[:, 0], batch_t=x_batch[:, 1], batch_r=x_batch[:, 2])
            for i, _pred in enumerate(prediction.reshape([-1]).astype(int).tolist()):
                _threshold = rel_threshold[x_batch[i][2]]  # 越小越相似，为正样例
                pred = 1 if _pred < _threshold else 0
                y_pred.append(pred)
            y_batch[y_batch <= 0] = 0
            y_true.extend(y_batch.reshape([-1]).tolist())
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1


def get_rank_hit_metrics(y_id, pred_ids):
    """
    :type y_id: int
    :type pred_ids: list
    """
    ranking = pred_ids.index(y_id) + 1
    _matrics = {
        'MRR': 1.0 / ranking,
        'MR': float(ranking),
        'HITS@1': 1.0 if ranking <= 1 else 0.0,
        'HITS@3': 1.0 if ranking <= 3 else 0.0,
        'HITS@10': 1.0 if ranking <= 10 else 0.0}
    return _matrics

# def get_binary_aprf(y_batch, prediction, threshold=0.5):
#     """accuracy, precision, recall, f1 , APRF"""
#     prediction = prediction.copy()
#     prediction[prediction > threshold] = 1
#     prediction[prediction <= threshold] = 0
#     y_pred = prediction.reshape([-1]).astype(int).tolist()
#     # y_batch[y_batch == -1] = 0
#     y_batch[y_batch <= 0] = 0
#     y_true = y_batch.reshape([-1]).tolist()
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     # import ipdb
#     # ipdb.set_trace()
#     return accuracy, precision, recall, f1

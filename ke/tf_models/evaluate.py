from typing import List

import numpy as np
import tensorflow as tf

from ke.data_helper import DataHelper
from ke.evaluate.rank_metrics import RankMetrics
from ke.tf_models.model_utils import Saver, session_conf
from ke.tf_models.train import get_metrics


class Prediction(object):
    def __init__(self, model_name, data_set):
        self.model_name = model_name
        self.rank_metrics = RankMetrics()
        self.data_helper = DataHelper(data_set=data_set)
        self.entity_nums = len(self.data_helper.entity2id)
        self.relation_nums = len(self.data_helper.relation2id)
        self.load_model()

    def load_model(self):
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session(config=session_conf, graph=graph)
            Saver(self.model_name).load_model(self.sess)
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
            self.input_y = graph.get_operation_by_name("input_y").outputs[0]
            self.prediction = graph.get_operation_by_name("prediction").outputs[0]

    def test_step(self, batch_h: List, batch_t: List, batch_r: List):
        """ 用于预测
        :param batch_h: [0,6,3,...,h_id]
        :param batch_t: [0,6,3,...,t_id]
        :param batch_r: [0,6,3,...,r_id]
        :return:
        """
        x = np.asarray(list(zip(batch_h, batch_t, batch_r)))
        prediction = self.sess.run(self.prediction, feed_dict={self.input_x: x})
        return prediction

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
        predictions = self.test_step(test_h, test_t, test_r)
        head_ids = predictions.reshape(-1).argsort()[::-1].tolist()
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
        predictions = self.test_step(test_h, test_t, test_r)
        tail_ids = predictions.reshape(-1).argsort()[::-1].tolist()
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
        predictions = self.test_step(test_h, test_t, test_r)
        relations = predictions.reshape(-1).argsort()[::-1].tolist()
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
        prediction = self.test_step([h], [t], [r])
        return prediction

    def get_metrics(self, y_id, pred_ids):
        mr = self.rank_metrics.mr(y_id=y_id, pred_ids=pred_ids)
        mrr = self.rank_metrics.mrr(y_id=y_id, pred_ids=pred_ids)
        hit_1 = self.rank_metrics.hit_k_count(y_ids=[y_id], pred_ids=pred_ids, k=1)
        hit_3 = self.rank_metrics.hit_k_count(y_ids=[y_id], pred_ids=pred_ids, k=3)
        hit_10 = self.rank_metrics.hit_k_count(y_ids=[y_id], pred_ids=pred_ids, k=10)
        return mr, mrr, hit_1, hit_3, hit_10

    def test_link_prediction(self):
        """
        链接预测，预测头实体或尾实体
        """
        mrs = []
        mrrs = []
        hit_1s = []
        hit_3s = []
        hit_10s = []
        for h, t, r in self.data_helper.data["test"]:
            pred_head_ids = self.predict_head_entity(t, r)
            mr, mrr, hit_1, hit_3, hit_10 = self.get_metrics(y_id=h, pred_ids=pred_head_ids)
            mrs.append(mr), mrrs.append(mrr), hit_1s.append(hit_1), hit_3s.append(hit_3), hit_10s.append(hit_10)
            pred_tail_ids = self.predict_tail_entity(h, r)
            mr, mrr, hit_1, hit_3, hit_10 = self.get_metrics(y_id=t, pred_ids=pred_tail_ids)
            mrs.append(mr), mrrs.append(mrr), hit_1s.append(hit_1), hit_3s.append(hit_3), hit_10s.append(hit_10)
        mr = np.mean(mrs)
        mrr = np.mean(mrrs)
        hit_1 = np.mean(hit_1s)
        hit_3 = np.mean(hit_3s)
        hit_10 = np.mean(hit_10s)
        # logging.info("mr:{:.4f}, mrr:{:.4f}, hit_1:{:.4f}, hit_3:{:.4f}, hit_10:{:.4f}".format(
        #     mr, mrr, hit_1, hit_3, hit_10))
        return mr, mrr, hit_1, hit_3, hit_10

    def test_triple_classification(self):
        accs = []
        precissions = []
        recalls = []
        f1s = []
        for x_batch, y_batch in self.data_helper.batch_iter(data_type="test",
                                                            batch_size=128,
                                                            epoch_nums=1):
            prediction = self.sess.run(self.prediction, feed_dict={self.input_x: x_batch,
                                                                   self.input_y: y_batch})
            accuracy, precision, recall, f1 = get_metrics(prediction, y_batch)
            accs.append(accuracy)
            precissions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        accuracy = np.mean(accs)
        precision = np.mean(precissions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
        return accuracy, precision, recall, f1

from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ke.config import Config
from ke.data_helper import DataHelper
from ke.evaluate.rank_metrics import RankMetrics
from ke.tf_models.model_utils.conf import session_conf
from ke.tf_models.model_utils.saver import Saver


class Predictor(object):
    def __init__(self, model_name, data_set, sess=None):
        self.model_name = model_name
        self.data_set = data_set
        self.rank_metrics = RankMetrics()
        self.data_helper = DataHelper(data_set=data_set)
        self.entity_nums = len(self.data_helper.entity2id)
        self.relation_nums = len(self.data_helper.relation2id)
        self.trasx_models = ["TransE"]
        self.load_model()

    def load_model(self):
        # if self.model_name == "TransformerKB":
        #     return
        graph = tf.Graph()
        self.sess = tf.Session(config=session_conf, graph=graph)
        with graph.as_default():  # self 无法load TransformerKB
            saver = Saver(self.model_name, relative_dir=self.data_set, allow_empty=True)
            saver.restore_model(self.sess)
            if self.model_name in ["ConvKB", "TransformerKB"]:  # ConvKB,TransformerKB
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                self.input_y = graph.get_operation_by_name("input_y").outputs[0]
            elif self.model_name in self.trasx_models:
                self.predict_h = graph.get_operation_by_name("predict_h").outputs[0]
                self.predict_t = graph.get_operation_by_name("predict_t").outputs[0]
                self.predict_r = graph.get_operation_by_name("predict_r").outputs[0]
            else:
                raise ValueError(self.model_name)
            self.prediction = graph.get_operation_by_name("predict").outputs[0]

    def predict(self, batch_h: List, batch_t: List, batch_r: List):
        """ 用于预测
        :param batch_h: [0,6,3,...,h_id]
        :param batch_t: [0,6,3,...,t_id]
        :param batch_r: [0,6,3,...,r_id]
        :return:
        """
        # if self.model_name == "TransformerKB":
        #     from ke.tf_models.models import TransformerKB
        #     num_ent_tags = len(self.data_helper.entity2id)
        #     num_rel_tags = len(self.data_helper.relation2id)
        #     model = TransformerKB(self.data_set, num_ent_tags, num_rel_tags, embedding_dim=Config.ent_emb_dim)
        #     graph = tf.Graph()
        #     with graph.as_default(), tf.Session(config=session_conf, graph=graph) as sess:  # self 无法load TransformerKB
        #         sess.run(tf.global_variables_initializer())
        #         model.saver.restore_model(sess, fail_ok=False)
        #         x = np.asarray(list(zip(batch_h, batch_r, batch_t)))
        #         prediction = sess.run(model.predict, feed_dict={model.input_x: x})
        #     return prediction
        if self.model_name in ["ConvKB", "TransformerKB"]:  # ConvKB,TransformerKB
            x = np.asarray(list(zip(batch_h, batch_r, batch_t)))
            prediction = self.sess.run(self.prediction, feed_dict={self.input_x: x})
            return prediction
        elif self.model_name in self.trasx_models:  # TransX
            x = np.asarray(list(zip(batch_h, batch_t, batch_r)))
            input_size = x.shape[0]
            pad = np.zeros(Config.batch_size - input_size)
            batch_h = np.concatenate([batch_h, pad], axis=0)
            batch_t = np.concatenate([batch_t, pad], axis=0)
            batch_r = np.concatenate([batch_r, pad], axis=0)
            prediction = self.sess.run(self.prediction, feed_dict={self.predict_h: batch_h,
                                                                   self.predict_t: batch_t,
                                                                   self.predict_r: batch_r})
            return prediction[:input_size]
        else:
            raise ValueError(self.model_name)

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
        predictions = self.predict(test_h, test_t, test_r)
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
        predictions = self.predict(test_h, test_t, test_r)
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
        prediction = self.predict([h], [t], [r])
        return prediction


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


class Evaluator(Predictor):
    def __init__(self, model_name, data_set, data_type="test", sess=None):
        super().__init__(model_name, data_set, sess)
        self.data_type = data_type

    def test(self):
        ranks = []
        ranks_left = []
        ranks_right = []
        hits = {1: [], 3: [], 10: []}
        hits_left = {1: [], 3: [], 10: []}
        hits_right = {1: [], 3: [], 10: []}
        for h, t, r in self.data_helper.data[self.data_type]:
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
        for h, t, r in self.data_helper.data[self.data_type]:
            pred_head_ids = self.predict_head_entity(t, r)
            _metrics = get_rank_hit_metrics(y_id=h, pred_ids=pred_head_ids)
            metrics_li.append(_metrics)
            pred_tail_ids = self.predict_tail_entity(h, r)
            _metrics = get_rank_hit_metrics(y_id=t, pred_ids=pred_tail_ids)
            metrics_li.append(_metrics)
        metrics = {}
        for metric_name in metrics_li[0].keys():
            metrics[metric_name] = sum([_metrics[metric_name] for _metrics in metrics_li]) / len(metrics_li)
        mr, mrr, hit_1, hit_3, hit_10 = (metrics["MR"], metrics["MRR"],
                                         metrics["HITS@10"], metrics["HITS@3"], metrics["HITS@1"])
        # logging.info("mr:{:.4f}, mrr:{:.4f}, hit_1:{:.4f}, hit_3:{:.4f}, hit_10:{:.4f}".format(
        #     mr, mrr, hit_1, hit_3, hit_10))
        return mr, mrr, hit_1, hit_3, hit_10

    def test_triple_classification(self):
        y_true = []
        y_pred = []
        for x_batch, y_batch in self.data_helper.batch_iter(data_type=self.data_type, batch_size=128, mode="htr"):
            prediction = self.predict(batch_h=x_batch[:, 0], batch_t=x_batch[:, 1], batch_r=x_batch[:, 2])
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            y_pred.extend(prediction.reshape([-1]).astype(int).tolist())
            y_batch[y_batch == -1] = 0
            y_true.extend(y_batch.reshape([-1]).tolist())
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

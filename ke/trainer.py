import logging

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import Config
from ke.data_helper import DataHelper
from ke.evaluator import Evaluator, get_rank_hit_metrics
from ke.models import (Analogy, ComplEx, DistMult, HolE, RESCAL,
                       TransD, TransE, TransH, TransR,
                       ConvKB, TransformerKB)


class Trainer(object):
    def __init__(self, model_name, data_set, min_num_epoch=Config.min_epoch_nums):
        self.model_name = model_name
        self.data_set = data_set
        self.min_num_epoch = min_num_epoch
        self.data_helper = DataHelper(data_set, model_name)
        # evaluate
        self.evaluator = Evaluator(model_name=self.model_name, data_set=self.data_set, load_model=False)
        self.min_loss = 100000
        self.best_val_mrr = 0
        self.patience_counter = 0

    def get_model(self):
        num_ent_tags = len(self.data_helper.entity2id)
        num_rel_tags = len(self.data_helper.relation2id)
        Model = {"Analogy": Analogy, "ComplEx": ComplEx, "DistMult": DistMult, "HolE": HolE, "RESCAL": RESCAL,
                 "TransD": TransD, "TransE": TransE, "TransH": TransH, "TransR": TransR,
                 "ConvKB": ConvKB, "TransformerKB": TransformerKB}[self.model_name]
        model = Model(self.data_set, num_ent_tags, num_rel_tags)
        model._build()
        return model

    def check_loss_save(self, sess, model, global_step, loss):
        if loss <= self.min_loss:
            model.saver.save_model(sess, global_step=global_step, loss=loss, mode="min_loss")
            self.min_loss = loss
        else:
            self.patience_counter += 1

    def check_save_mrr(self, sess, model, global_step):
        self.evaluator.set_model(sess=sess, model=model)
        mr, mrr, hit_10, hit_3, hit_1 = self.evaluator.test_link_prediction(
            data_type="valid", _tqdm=len(self.data_helper.entity2id) > 1000)
        # rank_metrics = "\n*model:{} {} valid, mrr:{:.3f}, mr:{:.3f}, hit_10:{:.3f}, hit_3:{:.3f}, hit_1:{:.3f}\n".format(
        #     self.model_name, self.data_set, mrr, mr, hit_10, hit_3, hit_1)
        if mrr >= self.best_val_mrr:
            ckpt_path = model.saver.save_model(sess, global_step=global_step, accuracy=mrr, mode="max_acc")
            # logging.info(rank_metrics)
            # print(rank_metrics)
            logging.info("get best mrr: {:.3f}, save to : {}".format(mrr, ckpt_path))
            self.best_val_mrr = mrr
        else:
            self.patience_counter += 1
        return mr, mrr, hit_10, hit_3, hit_1

    def run(self):
        logging.info("{} {} start train ...".format(self.model_name, self.data_set))
        graph = tf.Graph()
        sess = tf.Session(config=Config.session_conf, graph=graph)
        with graph.as_default(), sess.as_default():
            # get model
            model = self.get_model()
            sess.run(tf.global_variables_initializer())
            # if not Path(model.saver.get_model_path(mode=Config.load_model_mode, verbose=False)).is_file():
            #     print("* not found saved model:{}".format(self.model_name))
            #     ckpt_save_path = model.saver.save_model(sess, global_step=0, accuracy=0.0, loss=100.0,
            #                                             mode=Config.load_model_mode)
            #     print("save init state to : {}".format(ckpt_save_path))  # 0 step state save test file
            if Config.load_pretrain:  # 断点续训
                model_path = model.saver.restore_model(sess, fail_ok=True)
                if model_path:
                    print("* Model load from file: {}".format(model_path))
                else:
                    print("* not found saved model:{}".format(self.model_name))
            per_epoch_step = len(self.data_helper.data["train"]) // Config.batch_size // 2  # 正负样本
            global_step = sess.run(model.global_step)
            start_epoch_num = global_step // per_epoch_step  # 已经训练过多少epoch
            for epoch_num in trange(1, max(self.min_num_epoch, Config.max_epoch_nums) + 1,
                                    desc="{} {} train epoch ".format(self.model_name, self.data_set)):
                if epoch_num <= start_epoch_num:
                    continue
                losses = []
                for x_batch, y_batch in self.data_helper.batch_iter(data_type="train",
                                                                    batch_size=Config.batch_size, _shuffle=True):
                    _, pred, global_step, loss = sess.run(
                        [model.train_op, model.predict, model.global_step, model.loss],
                        feed_dict={model.input_x: x_batch,
                                   model.input_y: y_batch,
                                   model.dropout_keep_prob: Config.dropout_keep_prob})
                    if global_step % Config.check_step == 0:
                        self.evaluator.set_model(sess, model)
                        metrics = self.evaluator.evaluate_metrics(x_batch.tolist(), _tqdm=False)
                        mr, mrr = metrics["ave"]["MR"], metrics["ave"]["MRR"]
                        hit_10, hit_3, hit_1 = metrics["ave"]["Hit@10"], metrics["ave"]["Hit@3"], metrics["ave"][
                            "Hit@1"]
                        logging.info("{} {} train epoch_num: {}, global_step: {}, loss: {:.3f}, "
                                     "mr: {:.3f}, mrr: {:.3f}, Hit@10: {:.3f}, Hit@3: {:.3f}, Hit@1: {:.3f}".format(
                            self.data_set, self.model_name, epoch_num, global_step, loss,
                            mr, mrr, hit_10, hit_3, hit_1))
                        # logging.info(" step:{}, loss: {:.3f}".format(global_step, loss))
                    # predict = sess.run(model.predict, feed_dict={model.input_x: x_batch, model.input_y: y_batch})
                    losses.append(loss)
                self.check_loss_save(sess, model, global_step, loss)
                # if epoch_num > self.min_num_epoch:
                mr, mrr, hit_10, hit_3, hit_1 = self.check_save_mrr(sess, model, global_step)
                logging.info("{} {} valid epoch_num: {}, global_step: {}, loss: {:.3f}, "
                             "mr: {:.3f}, mrr: {:.3f}, hit_10: {:.3f}, hit_3: {:.3f}, hit_1: {:.3f}".format(
                    self.data_set, self.model_name, epoch_num, global_step, np.mean(losses),
                    mr, mrr, hit_10, hit_3, hit_1))
                model.saver.save_model(sess, global_step=global_step, loss=np.mean(losses), mode="max_step")
                logging.info("epoch {} end  ...\n------------------------------\n\n".format(epoch_num))
                # Early stopping and logging best f1
                if self.patience_counter >= Config.patience_num and epoch_num > self.min_num_epoch:
                    logging.info("{} {}, Best val f1: {:.3f} best loss:{:.3f}".format(
                        self.data_set, self.model_name, self.best_val_mrr, self.min_loss))
                    break

# def test(self, sess, model, global_step, loss, test_link_predict=False, test_triple_classification=False):
#
#     if loss <= self.min_loss:
#         model.saver.save_model(sess, global_step=global_step, loss=loss, mode="min_loss")
#         if loss - self.min_loss < Config.patience:
#             self.patience_counter += 1
#         else:
#             self.patience_counter = 0
#         self.min_loss = loss
#     else:
#         self.patience_counter += 1
#
#     # if self.evaluator is None:
#     #     self.evaluator = Evaluator(model_name=self.model_name, data_set=self.data_set, data_type="valid")
#
#     # if test_link_predict:
#     #     mr, mrr, hit_10, hit_3, hit_1 = self.evaluator.test_link_prediction()
#     #     rank_metrics = "\n*model:{}, mrr:{:.3f}, mr:{:.3f}, hit_10:{:.3f}, hit_3:{:.3f}, hit_1:{:.3f}\n".format(
#     #         self.model_name, mrr, mr, hit_10, hit_3, hit_1)
#     #     logging.info(rank_metrics)
#     #     print(rank_metrics)
#     #     if mr > self.best_mr:
#     #         model.saver.save_model(sess, global_step=global_step, accuracy=mrr)
#     #     self.best_mr = mr
#     #
#     # if test_triple_classification:
#     #     acc, precision, recall, f1 = self.evaluator.test_triple_classification()
#     #     logging.info("valid acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
#     #         acc, precision, recall, f1))
#     #     if f1 > self.best_val_f1:
#     #         model_path = model.saver.save_model(sess, global_step=global_step, loss=loss)
#     #         logging.info("** - Found new best F1 ,save to model_path: {}".format(model_path))
#     #         if f1 - self.best_val_f1 < Config.patience:
#     #             self.patience_counter += 1
#     #         else:
#     #             self.patience_counter = 0
#     #         self.best_val_f1 = f1
#     #     else:
#     #         self.patience_counter += 1

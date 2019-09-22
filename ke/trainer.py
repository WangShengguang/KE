import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import Config
from ke.data_helper import DataHelper
from ke.models import (Analogy, ComplEx, DistMult, HolE, RESCAL,
                       TransD, TransE, TransH, TransR,
                       ConvKB, TransformerKB)


class Trainer(object):
    def __init__(self, model_name, data_set, min_num_epoch=Config.min_epoch_nums):
        self.model_name = model_name
        self.data_set = data_set
        self.min_num_epoch = min_num_epoch
        self.data_helper = DataHelper(data_set)
        # evaluate
        # self.evaluator = None
        self.min_loss = 100000
        self.best_mr = 0.0
        self.best_val_f1 = 0
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

    def test(self, sess, model, global_step, loss):
        if loss <= self.min_loss:
            model.saver.save_model(sess, global_step=global_step, loss=loss, mode="min_loss")
            if loss - self.min_loss < Config.patience:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
            self.min_loss = loss
        else:
            self.patience_counter += 1

    def run(self):
        logging.info("{} {} start train ...".format(self.model_name, self.data_set))
        graph = tf.Graph()
        sess = tf.Session(config=Config.session_conf, graph=graph)
        with graph.as_default(), sess.as_default():
            # get model
            model = self.get_model()
            sess.run(tf.global_variables_initializer())
            if not Path(model.saver.get_model_path(mode=Config.load_model_mode) + ".meta").is_file():
                model.saver.save_model(sess, global_step=0, loss=100.0, mode="max_step")  # 0 step state save test file
            elif Config.load_pretrain:  # 断点续训
                model_path = model.saver.restore_model(sess, fail_ok=True)
                if model_path:
                    print("* Model load from file: {}".format(model_path))
            for epoch_num in trange(1, max(self.min_num_epoch, Config.max_epoch_nums) + 1,
                                    desc="{} {} train epoch ".format(self.model_name, self.data_set)):
                losses = []
                for x_batch, y_batch in self.data_helper.batch_iter(data_type="train",
                                                                    batch_size=Config.batch_size, _shuffle=True):
                    _, global_step, loss = sess.run([model.train_op, model.global_step, model.loss],
                                                    feed_dict={model.input_x: x_batch,
                                                               model.input_y: y_batch,
                                                               model.dropout_keep_prob: Config.dropout_keep_prob})
                    if global_step % Config.save_step == 0:
                        logging.info(" step:{}, loss: {:.4f}".format(global_step, loss))
                        self.test(sess, model, global_step, loss)
                    # predict = sess.run(model.predict, feed_dict={model.input_x: x_batch, model.input_y: y_batch})
                    losses.append(loss)
                self.test(sess, model, global_step, loss)
                model.saver.save_model(sess, global_step=global_step, loss=np.mean(losses), mode="max_step")
                logging.info("epoch {}, loss:{:.4f} ...\n".format(epoch_num, np.mean(losses)))
                # Early stopping and logging best f1
                if self.patience_counter >= Config.patience_num and epoch_num > self.min_num_epoch:
                    logging.info("{} {}, Best val f1: {:.4f} best loss:{:.4f}".format(
                        self.model_name, self.data_set, self.best_val_f1, self.min_loss))
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
#     #     rank_metrics = "\n*model:{}, mrr:{:.4f}, mr:{:.4f}, hit_10:{:.4f}, hit_3:{:.4f}, hit_1:{:.4f}\n".format(
#     #         self.model_name, mrr, mr, hit_10, hit_3, hit_1)
#     #     logging.info(rank_metrics)
#     #     print(rank_metrics)
#     #     if mr > self.best_mr:
#     #         model.saver.save_model(sess, global_step=global_step, accuracy=mrr)
#     #     self.best_mr = mr
#     #
#     # if test_triple_classification:
#     #     acc, precision, recall, f1 = self.evaluator.test_triple_classification()
#     #     logging.info("valid acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
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

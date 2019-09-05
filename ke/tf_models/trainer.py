import logging
from pathlib import Path

import tensorflow as tf
from tqdm import trange

from ke.config import Config
from ke.data_helper import DataHelper
from ke.tf_models.evaluator import Evaluator
from ke.tf_models.model_utils.conf import session_conf
from ke.tf_models.models import ConvKB, TransE, TransformerKB, other_models


class Trainer(object):
    def __init__(self, model_name, data_set):
        self.model_name = model_name
        self.data_set = data_set
        self.data_helper = DataHelper(data_set)
        # evaluate
        self.evaluator = None
        self.best_loss = 100000
        self.best_mr = 0.0
        self.best_val_f1 = 0
        self.patience_counter = 0

    def get_model(self):
        num_ent_tags = len(self.data_helper.entity2id)
        num_rel_tags = len(self.data_helper.relation2id)
        if self.model_name == "ConvKB":
            model = ConvKB(self.data_set, num_ent_tags, num_rel_tags, Config.ent_emb_dim, Config.rel_emb_dim)
        elif self.model_name == "TransformerKB":
            model = TransformerKB(self.data_set, num_ent_tags, num_rel_tags, embedding_dim=Config.ent_emb_dim)
        elif self.model_name == "TransE":
            model = TransE(self.data_set, num_ent_tags, num_rel_tags)
        else:
            raise ValueError(self.model_name)
        return model

    def test(self, sess, model, global_step, loss):
        # with torch.no_grad():  # 适用于测试阶段，不需要反向传播
        if self.evaluator is None:
            self.evaluator = Evaluator(model_name=self.model_name, data_set=self.data_set, data_type="valid")
        test_link_predict = False
        if test_link_predict:
            mr, mrr, hit_10, hit_3, hit_1 = self.evaluator.test_link_prediction()
            rank_metrics = "\n*model:{}, mrr:{:.4f}, mr:{:.4f}, hit_10:{:.4f}, hit_3:{:.4f}, hit_1:{:.4f}\n".format(
                self.model_name, mrr, mr, hit_10, hit_3, hit_1)
            logging.info(rank_metrics)
            print(rank_metrics)
            if mr > self.best_mr:
                model.saver.save_model(sess, global_step=global_step, accuracy=mrr)
            self.best_mr = mr
        test_triple_classification = False
        if test_triple_classification:
            acc, precision, recall, f1 = self.evaluator.test_triple_classification()
            logging.info("valid acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                acc, precision, recall, f1))
            if f1 > self.best_val_f1:
                model_path = model.saver.save_model(sess, global_step=global_step, loss=loss)
                logging.info("** - Found new best F1 ,save to model_path: {}".format(model_path))
                if f1 - self.best_val_f1 < Config.patience:
                    self.patience_counter += 1
                else:
                    self.patience_counter = 0
                self.best_val_f1 = f1
            else:
                self.patience_counter += 1
        if loss < self.best_loss:
            model.saver.save_model(sess, global_step=global_step, loss=loss)
            if loss - self.best_loss < Config.patience:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
            self.best_loss = loss
        else:
            self.patience_counter += 1

    def run(self):
        logging.info("start ... ")
        graph = tf.Graph()
        sess = tf.Session(config=session_conf, graph=graph)
        with graph.as_default(), sess.as_default():
            # get model
            model = self.get_model()
            sess.run(tf.global_variables_initializer())
            if not Path(model.saver.get_model_path(mode=Config.load_model_mode) + ".meta").is_file():
                model.saver.save_model(sess, global_step=0, loss=100.0)  # 0 step state save as init test file
            elif Config.load_pretrain:  # 断点续训
                model_path = model.saver.restore_model(sess, fail_ok=True)
                if model_path:
                    print("* Model load from file: {}".format(model_path))
            logging.info("{} start train ...".format(self.model_name))
            # mode = "concat" if self.model_name in other_models else "mix"
            for epoch_num in trange(Config.epoch_nums, desc="train epoch num"):
                for x_batch, y_batch in self.data_helper.batch_iter(data_type="train",
                                                                    batch_size=Config.batch_size,
                                                                    neg_label=-1.0, _shuffle=True):
                    _, global_step, loss = sess.run([model.train_op, model.global_step, model.loss],
                                                    feed_dict={model.input_x: x_batch,
                                                               model.input_y: y_batch})
                    if global_step % Config.save_step == 0:
                        logging.info(" step:{}, loss: {:.4f}".format(global_step, loss))
                    # predict = sess.run(model.predict, feed_dict={model.input_x: x_batch, model.input_y: y_batch})
                    # import ipdb
                    # ipdb.set_trace()
                self.test(sess, model, global_step, loss)
                logging.info("epoch {} end ...".format(epoch_num))
                # Early stopping and logging best f1
                if (self.patience_counter >= Config.patience_num and epoch_num > Config.min_epoch_nums) \
                        or epoch_num == Config.max_epoch_nums:
                    logging.info("{}, Best val f1: {:.4f}".format(self.model_name, self.best_val_f1))
                    break

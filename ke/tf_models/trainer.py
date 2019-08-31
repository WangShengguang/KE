import logging

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from ke.config import Config
from ke.data_helper import DataHelper
from ke.tf_models.model_utils.conf import session_conf
from ke.tf_models.models import ConvKB, TransE, TransformerKB


class Trainer(object):
    def __init__(self, model_name, data_set):
        self.model_name = model_name
        self.data_set = data_set
        self.data_helper = DataHelper(data_set)
        # evaluate
        self.best_loss = 100000
        self.patience_counter = 0

    def get_model(self):
        num_ent_tags = len(self.data_helper.entity2id)
        num_rel_tags = len(self.data_helper.relation2id)
        if self.model_name == "ConvKB":
            model = ConvKB(
                self.data_set, num_ent_tags, num_rel_tags,
                embedding_size=Config.ent_emb_dim,
                filter_sizes=[1],
                num_filters=500)
        elif self.model_name == "TransformerKB":
            model = TransformerKB(self.data_set, num_ent_tags, num_rel_tags, embedding_dim=Config.ent_emb_dim)
        elif self.model_name == "TransE":
            model = TransE(self.data_set, num_ent_tags, num_rel_tags)
        else:
            raise ValueError(self.model_name)
        return model

    def check(self, loss):
        if loss < self.best_loss:
            if self.best_loss - loss < Config.patience:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
            self.best_loss = loss
            return True
        else:
            self.patience_counter += 1
        return False

    def evaluate(self, prediction, y_batch):
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        y_pred = prediction.reshape([-1]).astype(int).tolist()
        y_batch[y_batch == -1] = 0
        y_true = y_batch.reshape([-1]).tolist()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    def run(self):
        logging.info("start ... ")
        graph = tf.Graph()
        sess = tf.Session(config=session_conf, graph=graph)
        with graph.as_default(), sess.as_default():
            # get model
            model = self.get_model()
            sess.run(tf.global_variables_initializer())
            if Config.load_pretrain:  # 断点续训
                model_path = model.saver.restore_model(sess, fail_ok=True)
                print("* Model load from file: {}".format(model_path))
            logging.info("{} start train ...".format(self.model_name))
            for epoch in range(Config.epoch_nums):
                for x_batch, y_batch in self.data_helper.batch_iter(data_type="train",
                                                                    batch_size=Config.batch_size,
                                                                    neg_label=-1):
                    assert not np.any(np.isnan(x_batch))
                    _, global_step, loss = sess.run([model.train_op, model.global_step, model.loss],
                                                    feed_dict={model.input_x: x_batch,
                                                               model.input_y: y_batch})
                    logging.info(" step:{}, loss: {:.6f}".format(global_step, loss))
                    # import ipdb
                    # ipdb.set_trace()
                    # pos_h,h, t, r = sess.run([model.pos_h,model.h, model.t, model.r],
                    #                    feed_dict={model.input_x: x_batch, model.input_y: y_batch})
                    # print(h, t, r)
                    # import ipdb
                    # ipdb.set_trace()
                    if global_step > 0 and global_step % Config.save_step == 0:
                        predict = sess.run(model.predict, feed_dict={model.input_x: x_batch, model.input_y: y_batch})
                        accuracy, precision, recall, f1 = self.evaluate(predict, y_batch)
                        _test_log = "* test acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                            accuracy, precision, recall, f1)
                        logging.info(_test_log)
                        if self.check(loss):
                            model.saver.save_model(sess, global_step=global_step, loss=loss)
                if self.check(loss):
                    model.saver.save_model(sess, global_step=global_step, loss=loss)
                # accuracy, precision, recall, f1 = get_metrics(prediction, y_batch)
                # logging.info("accuracy:{:.6f}, precision:{:.6f}, recall:{:.6f}, f1:{:.6f}".format(
                #     accuracy, precision, recall, f1))
                logging.info("epoch {} end ...".format(epoch))

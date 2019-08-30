import logging

import tensorflow as tf

from ke.config import Config
from ke.data_helper import DataHelper
from ke.tf_models.model_utils.conf import session_conf
from ke.tf_models.models import ConvKB, TransE


class Trainer(object):
    def __init__(self, model_name, data_set):
        self.model_name = model_name
        self.data_set = data_set
        self.data_helper = DataHelper(data_set)
        self.mode = "htr"  # head tail relation

    def get_model(self):
        num_ent_tags = len(self.data_helper.entity2id)
        num_rel_tags = len(self.data_helper.relation2id)
        if self.model_name == "ConvKB":
            vocab_size = len(self.data_helper.entity2id) + len(self.data_helper.relation2id)
            model = ConvKB(
                embedding_size=50,
                filter_sizes=[1],
                num_filters=500,
                vocab_size=vocab_size,
                data_set=self.data_set)
            self.mode = "hrt"
        elif self.model_name == "TransE":
            model = TransE(num_ent_tags=num_ent_tags, num_rel_tags=num_rel_tags, data_set=self.data_set)
        else:
            raise ValueError(self.model_name)
        return model

    def run(self):
        logging.info("start ... ")
        graph = tf.Graph()
        sess = tf.Session(config=session_conf, graph=graph)
        with graph.as_default(), sess.as_default():
            # get model
            model = self.get_model()
            sess.run(tf.global_variables_initializer())
            if Config.load_pretrain:  # 断点续训
                model.load(sess, fail_ok=True)
            logging.info("{} start train ...".format(self.model_name))
            for epoch in range(Config.epoch_nums):
                for x_batch, y_batch in self.data_helper.batch_iter(data_type="train",
                                                                    batch_size=Config.batch_size,
                                                                    mode=self.mode):
                    _, global_step, loss = sess.run([model.train_op, model.global_step, model.loss],
                                                    feed_dict={
                                                        model.input_x: x_batch,
                                                        model.input_y: y_batch})
                    logging.info(" step:{}, loss: {:.6f}".format(global_step, loss))
                    if global_step > 0 and global_step % Config.save_step == 0:
                        model.save(sess, loss=loss)
                    # accuracy, precision, recall, f1 = get_metrics(prediction, y_batch)
                    # logging.info("accuracy:{:.6f}, precision:{:.6f}, recall:{:.6f}, f1:{:.6f}".format(
                    #     accuracy, precision, recall, f1))
                logging.info("epoch {} end ...".format(epoch))

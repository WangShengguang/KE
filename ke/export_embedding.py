import json
import logging
import os

import tensorflow as tf

from config import EmbeddingExportConfig
from ke.utils.saver import Saver


class EmbeddingExporter(object):
    def __init__(self, data_set, model_name):
        self.data_set = data_set
        self.model_name = model_name
        self.config = EmbeddingExportConfig(data_set=data_set)
        self.embedding_json_path = os.path.join(self.config.embedding_dir, f"{model_name}.json")

    def export_embedding(self):
        graph = tf.Graph()
        sess = tf.Session(config=self.config.session_conf, graph=graph)
        with graph.as_default(), sess.as_default():  # self 无法load TransformerKB
            model_path = Saver(data_set=self.data_set, model_name=self.model_name).restore_model(sess)
            print("* load model path :{}".format(model_path))
            if self.model_name == "RESCAL":
                # 无需拆分 ent_emb和rel_emb，后续SimRank推荐 ent2id rel2id从datahelper获得，在其中处理之即可
                rel_embeddings = graph.get_operation_by_name("rel_matrices").outputs[0]
            else:
                rel_embeddings = graph.get_operation_by_name("rel_embeddings").outputs[0]
            ent_embeddings = graph.get_operation_by_name("ent_embeddings").outputs[0]

            ent_embeddings = sess.run(ent_embeddings)
            rel_embeddings = sess.run(rel_embeddings)
        with open(self.embedding_json_path, "w", encoding="utf-8") as f:
            json.dump({"ent_embeddings": ent_embeddings.tolist(), "rel_embeddings": rel_embeddings.tolist()},
                      f, indent=4, ensure_ascii=False)
        logging.info("embedding save to : {} ,ent_embeddings:{},rel_embeddings:{}".format(
            self.embedding_json_path, ent_embeddings.shape, rel_embeddings.shape))
        return self.embedding_json_path

    def load_embedding(self, reuse=False):
        if not reuse or not os.path.isfile(self.embedding_json_path):
            self.export_embedding()
        with open(self.embedding_json_path, 'r', encoding='utf-8') as f:
            embedding = json.load(f)
            entity_embedding = embedding['ent_embeddings']
            relation_embedding = embedding['rel_embeddings']
        return entity_embedding, relation_embedding

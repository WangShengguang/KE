import json
import logging
import os

import tensorflow as tf

from config import Config
from ke.utils.saver import Saver


class EmbeddingExporter(object):
    def __init__(self, data_set, model_name):
        self.data_set = data_set
        self.model_name = model_name
        self.embedding_json_path = Config.embedding_json_path_tmpl.format(data_set=data_set, model_name=model_name)
        os.makedirs(os.path.dirname(self.embedding_json_path), exist_ok=True)

    def export_embedding(self, ):
        graph = tf.Graph()
        sess = tf.Session(config=Config.session_conf, graph=graph)
        with graph.as_default(), sess.as_default():  # self 无法load TransformerKB
            model_path = Saver(self.model_name, relative_dir=self.data_set, allow_empty=True).restore_model(sess)
            print("* load model path :{}".format(model_path))
            if self.model_name == "ConvKB":
                ent_embeddings = graph.get_operation_by_name("ConvKB-W").outputs[0]
                rel_embeddings = ent_embeddings
            elif self.model_name == "TransformerKB":
                ent_embeddings = graph.get_operation_by_name("Tramsformer-W").outputs[0]
                rel_embeddings = ent_embeddings
            else:
                ent_embeddings = graph.get_operation_by_name("ent_embeddings").outputs[0]
                rel_embeddings = graph.get_operation_by_name("rel_embeddings").outputs[0]
            ent_embeddings = sess.run(ent_embeddings)
            rel_embeddings = sess.run(rel_embeddings)
        with open(self.embedding_json_path, "w", encoding="utf-8") as f:
            json.dump({"ent_embeddings": ent_embeddings.tolist(), "rel_embeddings": rel_embeddings.tolist()},
                      f, indent=4, ensure_ascii=False)
        logging.info("embedding save to : {} ,ent_embeddings:{},rel_embeddings:{}".format(
            self.embedding_json_path, ent_embeddings.shape, rel_embeddings.shape))
        return self.embedding_json_path

    def load_embedding(self):
        if not os.path.isfile(self.embedding_json_path):
            self.export_embedding()
        with open(self.embedding_json_path, 'r', encoding='utf-8') as f:
            embedding = json.load(f)
            entity_embedding = embedding['ent_embeddings']
            relation_embedding = embedding['rel_embeddings']
        return entity_embedding, relation_embedding

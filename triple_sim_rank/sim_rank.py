import json
import os
import random

import numpy as np
import pandas as pd

from config import Config
from ke.data_helper import DataHelper
from triple_sim_rank.export_embedding import EmbeddingExporter
from triple_sim_rank.utils import graph_embedding
from triple_sim_rank.utils import split_triple, get_hit_type


def cos_sim(vector1, vector2):
    vector_a = np.mat(vector1)
    vector_b = np.mat(vector2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    if np.isnan(sim):
        sim = 0.0
    return sim


def ranking(vector_data, case_id, choice_list):
    """
    :param vector_data: {case_id:[triple,triple]}
    :param case_id:
    :param choice_list: 选取相似度列表的 第n项 作为结果
    :return:
    """
    tmp_rank = {}
    vect1 = vector_data[case_id]
    for simi_case_id in vector_data.keys():
        if simi_case_id == case_id:
            continue
        vect2 = vector_data[simi_case_id]
        score = cos_sim(vect1, vect2)
        tmp_rank[simi_case_id] = score
    ranks = sorted(tmp_rank.items(), key=lambda item: item[1], reverse=True)
    similar_case = []
    for choice_case in choice_list:
        if choice_case > len(ranks):
            break
        simi_case_id, score = ranks[choice_case]
        similar_case.append([simi_case_id, score])
    return similar_case


class TripleSimRank(object):

    def __init__(self, data_set, model_name):
        self.data_set = data_set
        self.model_name = model_name
        self.data_helper = DataHelper(data_set=data_set, model_name=model_name)
        self.embedding_exporter = EmbeddingExporter(data_set=data_set, model_name=model_name)
        self.load_triples_cases()

    def load_triples_cases(self, case_nums=10):
        print("* {}".format(Config.cases_triples_json_tmpl.format(data_set=self.data_set)))
        with open(Config.cases_triples_json_tmpl.format(data_set=self.data_set), 'r', encoding='utf-8') as f:
            self.triples = json.load(f)
        case_list_file = Config.case_list_tmpl.format(data_set=self.data_set)
        if not os.path.isfile(case_list_file):
            case_list = [case_id for case_id in self.triples.keys() if self.triples[case_id]]
            choice_case_list = random.sample(case_list, min(case_nums, len(case_list)))
            with open(case_list_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(choice_case_list))
                # f.flush()
        with open(case_list_file, "r", encoding="utf-8") as f:
            self.case_list = [case_id.strip() for case_id in f.readlines()]  # 所有案由号

    def sim_rank(self, choice_list=(0, 1, 3, 4, 7, 8, 12, 13, 18, 19)):
        """
        :param model_name:  KE model's name
        :param choice_list: 选取相似度列表的 第n项 作为结果
        :return:
        """
        ent2id, rel2id = self.data_helper.entity2id, self.data_helper.relation2id
        entity_embedding, relation_embedding = self.embedding_exporter.load_embedding()
        print("{}, entity_embedding_size:{}, relation_embedding_size:{}".format(self.model_name, len(entity_embedding),
                                                                                len(relation_embedding)))
        car_hits_people, car_hits_car, car_hits_car_and_people, others = split_triple(self.triples)
        car_hits_people_vector = graph_embedding(car_hits_people, entity_embedding, relation_embedding, ent2id, rel2id)
        car_hits_car_vector = graph_embedding(car_hits_car, entity_embedding, relation_embedding, ent2id, rel2id)
        car_hits_car_and_people_vector = graph_embedding(car_hits_car_and_people, entity_embedding, relation_embedding,
                                                         ent2id, rel2id)
        others_vector = graph_embedding(others, entity_embedding, relation_embedding, ent2id, rel2id)
        rank_result = {}
        for case_id in self.case_list:
            trip = self.triples[case_id]
            car_people, car_car = get_hit_type(trip)
            if car_people and car_car:
                similar_case = ranking(car_hits_car_and_people_vector, case_id, choice_list)
            elif not car_people and car_car:
                similar_case = ranking(car_hits_car_vector, case_id, choice_list)
            elif car_people and not car_car:
                similar_case = ranking(car_hits_people_vector, case_id, choice_list)
            else:
                similar_case = ranking(others_vector, case_id, choice_list)
            rank_result[case_id] = similar_case
        out_rank_file = os.path.join(Config.sim_rank_dir, "{}.json".format(self.model_name))
        with open(out_rank_file, 'w', encoding='utf-8') as f:
            json.dump(rank_result, f, ensure_ascii=False, indent=4)
        return rank_result, out_rank_file


def create_all_sim_rank(data_set, model_names):
    choice_list = [0, 1, 3, 4, 7, 8, 12, 13, 18, 19]
    data_results = []
    for model_name in model_names:
        triple_sim_ranker = TripleSimRank(data_set=data_set, model_name=model_name)
        rank_result, out_rank_file = triple_sim_ranker.sim_rank(choice_list=choice_list)
        for case_id in triple_sim_ranker.case_list:
            simi_cases = rank_result[case_id]
            for i, simi_case in enumerate(simi_cases):
                data_results.append([case_id, simi_case, model_name, i + 1])
    column_names = ["案件案号", "类似案件案号", "算法", "算法给的相似性排序"]
    pd.DataFrame(data=data_results, columns=column_names).to_csv(Config.rank_result_csv)

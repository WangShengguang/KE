import json
import os
import random

import numpy as np
import pandas as pd

from config import SimRankConfig
from ke.data_helper import DataHelper
from ke.export_embedding import EmbeddingExporter
from triple_sim_rank.utils import graph_embedding
from triple_sim_rank.utils import split_triple, get_hit_type


def cos_sim(vector1, vector2):
    """
    https://blog.csdn.net/hqh131360239/article/details/79061535
    :param vector1:
    :param vector2:
    :return:
    """
    vector_a = np.mat(vector1)  # n,3,emb_dim
    vector_b = np.mat(vector2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    if np.isnan(sim):
        sim = 0.0
    return sim


def ranking(all_vector_data, query_case_id):
    """
    :param all_vector_data: {case_id:[[h,t,r],[h,t,r],...]}   head_emb,tail_emb,rel_emb
    :param query_case_id: 文章
    :return:
    """
    tmp_rank = {}
    vect1 = all_vector_data[query_case_id]
    for target_case_id in all_vector_data.keys():
        if target_case_id == query_case_id:
            continue
        vect2 = all_vector_data[target_case_id]
        score = cos_sim(vect1, vect2)
        tmp_rank[target_case_id] = score
    similar_case = sorted(tmp_rank.items(), key=lambda item: item[1], reverse=True)
    return similar_case


class TripleSimRank(object):

    def __init__(self, data_set, model_name):
        self.data_set = data_set
        self.model_name = model_name
        self.config = SimRankConfig(data_set=data_set, model_name=model_name)
        self.data_helper = DataHelper(data_set=data_set, model_name=model_name)
        self.embedding_exporter = EmbeddingExporter(data_set=data_set, model_name=model_name)
        with open(self.config.cases_triples_json, 'r', encoding='utf-8') as f:
            self.all_triples = json.load(f)
            print("*source cases_triples {}".format(self.config.cases_triples_json))
        self.all_triples_set = {case_id: set([tuple(triple) for triple in triples])
                                for case_id, triples in self.all_triples.items()}

    def get_query_case_ids(self, case_nums=10):
        if not os.path.isfile(self.config.case_list_txt):
            case_list = [case_id for case_id in self.all_triples.keys() if self.all_triples[case_id]]
            choice_case_list = random.sample(case_list, min(case_nums, len(case_list)))
            with open(self.config.case_list_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(choice_case_list))
                # f.flush()
        with open(self.config.case_list_txt, "r", encoding="utf-8") as f:
            query_case_ids = [case_id.strip() for case_id in f.readlines()]  # 所有案由号
        return query_case_ids

    def sim_rank(self, query_case_ids, k=10):
        """
        :param query_case_ids:
        :param k
        :return:
        """
        ent2id, rel2id = self.data_helper.entity2id, self.data_helper.relation2id
        entity_embedding, relation_embedding = self.embedding_exporter.load_embedding(reuse=False)
        print("{}, entity_embedding_size:{}, relation_embedding_size:{}".format(self.model_name, len(entity_embedding),
                                                                                len(relation_embedding)))
        car_hits_people, car_hits_car, car_hits_car_and_people, others = split_triple(self.all_triples)
        car_hits_people_vector = graph_embedding(car_hits_people, entity_embedding, relation_embedding, ent2id, rel2id)
        car_hits_car_vector = graph_embedding(car_hits_car, entity_embedding, relation_embedding, ent2id, rel2id)
        car_hits_car_and_people_vector = graph_embedding(car_hits_car_and_people, entity_embedding, relation_embedding,
                                                         ent2id, rel2id)
        others_vector = graph_embedding(others, entity_embedding, relation_embedding, ent2id, rel2id)
        rank_result = {case_id: {"sim_cases": [], "global_ave_score": -1, "full_score_count": 0}
                       for case_id in query_case_ids}
        for query_case_id in query_case_ids:
            trip = self.all_triples[query_case_id]
            car_people, car_car = get_hit_type(trip)
            if car_people and car_car:
                similar_case = ranking(car_hits_car_and_people_vector, query_case_id)
            elif not car_people and car_car:
                similar_case = ranking(car_hits_car_vector, query_case_id)
            elif car_people and not car_car:
                similar_case = ranking(car_hits_people_vector, query_case_id)
            else:
                similar_case = ranking(others_vector, query_case_id)
            rank_result[query_case_id]["global_ave_score"] = np.mean([score for case_id, score in similar_case])
            rank_result[query_case_id]["full_score_count"] = len([score for case_id, score in similar_case
                                                                  if score > 0.999])
            rank_result[query_case_id]["sim_cases"] = similar_case[:k]
        # with open(self.config.rank_result_csv, 'w', encoding='utf-8') as f:
        #     json.dump(rank_result, f, ensure_ascii=False, indent=4)
        return rank_result

    def tf_rank(self, query_case_id, k=10):
        """ Term Frequency 文档评率
        加一个指标，sim=分子是三元组共现的次数的对数/分母线上三元组总个数的对数 * 4
        线上，待查询文档 10，数据库12 ； 5/10 * 4
        0-4
        分子对数此分数与
        每个算法（所有文档）相似度的平均分
        满分次数 统计
        """
        # car_hits_people, car_hits_car, car_hits_car_and_people, others = split_triple(self.triples)
        sim_cases = []
        for target_case_id, target_triples_set in self.all_triples_set.items():
            if query_case_id == target_case_id:
                continue
            score = tf_sim(self.all_triples_set[query_case_id], target_triples_set)
            sim_cases.append((target_case_id, score))
        ranked_sim_cases = sorted(sim_cases, key=lambda x: x[1], reverse=True)
        global_ave_score = np.mean([score for case_id, score in ranked_sim_cases])
        full_score_count = len([score for case_id, score in ranked_sim_cases if score > 3.999])
        return ranked_sim_cases[:k], global_ave_score, full_score_count


def create_all_sim_rank(data_set, model_names, sim_k=10):
    data_results = []
    for model_name in model_names:
        triple_sim_ranker = TripleSimRank(data_set=data_set, model_name=model_name)
        query_case_ids = triple_sim_ranker.get_query_case_ids()
        cos_sim_rank_result = triple_sim_ranker.sim_rank(query_case_ids, k=sim_k)
        for query_case_id in query_case_ids:
            cos_rank_result = cos_sim_rank_result[query_case_id]
            # tf_sim_cases, tf_global_ave_score, tf_full_score_count = triple_sim_ranker.tf_rank(query_case_id, k=sim_k)
            for i, simi_case in enumerate(cos_rank_result["sim_cases"]):
                data_results.append([query_case_id, simi_case, model_name, i + 1,
                                     cos_rank_result["global_ave_score"], cos_rank_result["full_score_count"],
                                     # tf_sim_cases[i], tf_global_ave_score, tf_full_score_count
                                     ])
    column_names = ["案件案号", "类似案件案号", "算法", "算法给的相似性排序",
                    "cos全局平均", "cos满分个数",
                    # "Joint_类似案件案号", "Joint_全局平均", "Joint_满分个数"
                    ]
    rank_result_csv = SimRankConfig(data_set=data_set, model_name="all").rank_result_csv
    pd.DataFrame(data=data_results, columns=column_names).to_csv(rank_result_csv, index=False, encoding="utf-8-sig")
    print("* save to :{}".format(rank_result_csv))


def tf_sim(query_triples_set, target_triples_set):
    overlap_set = query_triples_set & target_triples_set
    score = len(overlap_set) / len(query_triples_set) * 4
    return score

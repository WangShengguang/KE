import numpy as np


def split_triple(triples):
    car_hits_people = {}
    car_hits_car = {}
    car_hits_car_and_people = {}
    others = {}
    for case_id in triples.keys():
        trip = triples[case_id]
        if len(trip) == 0:
            continue
        car_people, car_car = get_hit_type(trip)
        if car_people and car_car:
            car_hits_car_and_people[case_id] = trip
        elif not car_people and car_car:
            car_hits_car[case_id] = trip
        elif car_people and not car_car:
            car_hits_people[case_id] = trip
        else:
            others[case_id] = trip
    return car_hits_people, car_hits_car, car_hits_car_and_people, others


def get_hit_type(triple):
    # 判断车祸类型，1、机动车撞人（车撞非机动车）2、机动车撞机动车。3、两者都有。4、未发生碰撞
    car_people, car_car = False, False
    for trip in triple:
        ent1, rel, ent2 = trip[0], trip[1], trip[2]
        if rel != '发生事故':
            continue
        role = ent2.split('-')[-1]
        if '机动车' in role and '非机动车' not in role:
            car_car = True
        else:
            car_people = True
    return car_people, car_car


def graph_embedding(triple, ent_embedding, rel_embedding, ent2id, rel2id):
    graph_vector = {}
    ent_vecor_lens = len(ent_embedding[0])
    rel_vecor_lens = len(rel_embedding[0])
    for case_id in triple.keys():
        ent1_vector = np.zeros(ent_vecor_lens)
        rel_vector = np.zeros(rel_vecor_lens)
        ent2_vector = np.zeros(ent_vecor_lens)
        lens = len(triple[case_id])
        for trip in triple[case_id]:
            ent1 = trip[0]
            rel = trip[1]
            ent2 = trip[2]
            ent1_vector += np.array(ent_embedding[ent2id[ent1]])
            rel_vector += np.array(rel_embedding[rel2id[rel]])
            ent2_vector += np.array(ent_embedding[ent2id[ent2]])
        ent1_vector = np.array(ent1_vector / lens)
        rel_vector = np.array(rel_vector / lens)
        ent2_vector = np.array(ent2_vector / lens)
        vector = np.concatenate((ent1_vector, rel_vector, ent2_vector), axis=-1)
        graph_vector[case_id] = vector
    return graph_vector

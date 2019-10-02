# KE
Knowledge Embedding   ，知识编码

主要参考OpenKE，将其中C++部分用Python重新实现，过度抽象封装的部分进行重新封装 

---

## 1. 数据准备 
    
|input_data|out_data|
|---|---|
|benchmarks/dataset|output/dataset|

|数据集|说明|
|---|---|
|lawdata|原始标注集通过NERE后得到|
|lawdata_new|与lawdata同，避免生成数据覆盖而命名通过NERE后得到|
|traffic_all|未标注数据通过NERE后得到|


0.5738, mr:3.4272, hit_10:0.9417, hit_3:0.7039, hit_1:0.3932

## 2. Link Prediction

Link prediction aims to predict the missing h or t for a relation fact triple (h, r, t). In this task, for each position of missing entity, the system is asked to rank a set of candidate entities from the knowledge graph, instead of only giving one best result. For each test triple (h, r, t), we replace the head/tail entity by all entities in the knowledge graph, and rank these entities in descending order of similarity scores calculated by score function fr. we use two measures as our evaluation metric:

* ***MR*** : mean rank of the correct entities
* ***MRR***: the average of the reciprocal ranks of the correct entities
* ***Hit@N*** : proportion of correct entities in top-N ranked entities

### 2.1 lawdata（NERE训练用数据集）

从原始训练集在此通过NER&RE三元组抽取得到

| Datasets        | Number of Cases | Number of Entities | Number of Relations |
| --------------- | --------------- | ------------------ | ------------------- |
| Training Set    | 263            | 98                | 8                   |
| Development Set | 104             | 98                | 8                   |
| Test Set        | 103             | 98                | 8                   |

各模型在此数据集上的表现

 | Model         | MRR    | MR       | hit@10 | hit@3  | hit@1  |
 | ------------- | ------ | -------- | ------ | ------ | ------ |
 | Analogy       | 0.5673| 3.5388| 0.9223| 0.7136| 0.3689|
 | ComplEx       | 0.5753|3.5437|0.9223|0.7136|0.3835|
 | DistMult      | 0.5784|3.6602|0.9126|0.7039|0.4029 |
 | HolE          | 0.5343|4.1990|0.8786, hit_3:0.6602|0.3447|
 | RESCAL        | 0.5993|3.4126|0.9272, hit_3:0.7427|0.4175 |
 | TransD        | 0.3416|4.5583|0.9029, hit_3:0.6359|0.0097 |
 | TransE        | 0.3297|4.6214|0.9078|0.5874|0.0097 |
 | TransH        | 0.3433|4.4175|0.9126, hit_3:0.6068|0.0097 |
 | TransR        | 0.5583|3.5777|0.9272|0.6748|0.3689|
 | ConvKB        |0.5166|4.1553|0.9126|0.6117|0.3398|
 | TransformerKB | 0.5993|3.2427|0.9369|0.7573|0.4078 |


### 2.2 随机数据集
随机挑选的500篇文章通过NER&RE三元组抽取得到的数据集

| Datasets        | Number of Cases | Number of Entities | Number of Relations |
| --------------- | --------------- | ------------------ | ------------------- |
| Training Set    | 300            | 112                | 8                   |
| Development Set | 100             | 112                | 8                   |
| Test Set        | 100             | 112                | 8                   |

各模型在此数据集上的表现

 | Model         | MRR    | MR       | hit@10 | hit@3  | hit@1  |
 | ------------- | ------ | -------- | ------ | ------ | ------ |
 | Analogy       | 0.1648|37.0300|0.3550|0.1450|0.0800 |
 | ComplEx       | 0.1699|34.9000|0.3950|0.1700|0.0750 |
 | DistMult      |0.1816|30.9050|0.4150|0.2250|0.0650 |
 | HolE          | 0.2244|32.8200|0.4550|0.2600|0.1150 |
 | RESCAL        | 0.2077|29.4550|0.4650|0.2300|0.0950|
 | TransD        | 0.1383|28.8850|0.4150|0.1650|0.0100 |
 | TransE        | 0.1313|30.4500|0.3900|0.1500|0.0100 |
 | TransH        |0.1339|30.9900|0.3850|0.1600|0.0100 |
 | TransR        | 0.1656|27.6750|0.4350|0.1800|0.0450 |
 | ConvKB        | 0.2292|24.8350|0.5150|0.2700|0.1000 |
 | TransformerKB | 0.2446|23.3950|0.5300|0.3050|0.1050|


# tips
    data feed 
    多轮训练（断点续训）
    load max_step 而不是min_loss

## Debug
```bash
    ipython --pdb manage.py 
    python3 -m ipdb manage.py
```
    

## Reference
- [Awesome Knowledge Graph Embedding Approaches](https://gist.github.com/mommi84/07f7c044fa18aaaa7b5133230207d8d4)








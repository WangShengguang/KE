# KE
Knowledge Embedding，知识表示

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


参数设置，HolE,RESCAL,TransD,TransE,TransH,TransR使用hinge loss function,margin设置为1，
其余参数见下表 

|参数|value|
|---|---|
|batch_size|16|
|learning_rate | 0.0001|
|l2 regularizer lambda| 0.001|
|dropout_keep_prob|0.8|
|margin|1.0|
|ent_emb_dim|128|
|rel_emb_dim|128|

使用 Link Prediction任务和MR，MRR，Hit@N指标作为模型的评价方式

## 2. Link Prediction

Link prediction aims to predict the missing h or t for a relation fact triple (h, r, t). In this task, for each position of missing entity, the system is asked to rank a set of candidate entities from the knowledge graph, instead of only giving one best result. For each test triple (h, r, t), we replace the head/tail entity by all entities in the knowledge graph, and rank these entities in descending order of similarity scores calculated by score function fr. we use two measures as our evaluation metric:

* ***MR*** : mean rank of the correct entities
* ***MRR***: the average of the reciprocal ranks of the correct entities
* ***Hit@N*** : proportion of correct entities in top-N ranked entities

### 2.1 lawdata（NERE训练用数据集）

从训练集通过NER&RE三元组抽取和知识融合得到新的三元组，按3:1:1划分训练集、验证集、测试集

| Datasets        | Number of Cases | Number of Entities | Number of Relations |
| --------------- | --------------- | ------------------ | ------------------- |
| Training Set    | 263            | 98                | 8                   |
| Development Set | 104             | 98                | 8                   |
| Test Set        | 103             | 98                | 8                   |

各模型在此数据集上的表现

 | Model         | MR                | MRR       | hit@10 | hit@3  | hit@1  |
 | ------------- | ------ | -------- | ------ | ------ | ------ |
|Analogy        |       3.4029  |       0.5755  |       0.9369  |       0.7233  |       0.3835|  
|ComplEx        |       3.4515  |       0.6006  |       0.9320  |       0.7136  |       0.4320|  
|DistMult       |       3.9709  |       0.5444  |       0.9029  |       0.6602  |       0.3641|  
|HolE           |       4.3835  |       0.5328  |       0.8544  |       0.6602  |       0.3495|  
|RESCAL         |       3.5485  |       0.5689  |       0.9223  |       0.7136  |       0.3738|
|TransD         |       4.8883  |       0.3172  |       0.8835  |       0.5340  |       0.0097|  
|TransE         |       4.8058  |       0.3170  |       0.9078  |       0.5291  |       0.0097|  
|TransH         |       4.7379  |       0.3135  |       0.9078  |       0.5340  |       0.0097|  
|TransR         |       3.9854  |       0.4511  |       0.9320  |       0.6408  |       0.1990|  
|ConvKB         |       4.1602  |       0.5182  |       0.8981  |       0.6165  |       0.3398|  
| TransformerKB | 3.2427 |0.5993|0.9369|0.7573|0.4078 |

TransformerKB在各项指标均优于其余模型


### 2.2 随机数据集
随机挑选的500篇交通判决案例通过NER&RE三元组抽取和知识融合得到新的三元组，按3:1:1划分训练集、验证集、测试集


| Datasets        | Number of Cases | Number of Entities | Number of Relations |
| --------------- | --------------- | ------------------ | ------------------- |
| Training Set    | 300            | 111                | 8                   |
| Development Set | 100             | 111                | 8                   |
| Test Set        | 100             | 111                | 8                   |

各模型在此数据集上的表现

 | Model         | MR    | MRR       | hit@10 | hit@3  | hit@1  |
 | ------------- | ------ | -------- | ------ | ------ | ------ |
|Analogy        |       26.7300 |       0.1609  |       0.4100  |       0.1500  |       0.0600|  
|ComplEx        |       28.9050 |       0.1393  |       0.3950  |       0.1250  |       0.0350|  
|DistMult       |       29.3050 |       0.1396  |       0.3500  |       0.1400  |       0.0400|
|HolE           |       28.6300 |       0.1820  |       0.4500  |       0.2300  |       0.0650|
|RESCAL         |       31.4150 |       0.1129  |       0.3600  |       0.0750  |       0.0200|
|TransD         |       29.5000 |       0.1290  |       0.3650  |       0.1200  |       0.0200|  
|TransE         |       37.4650 |       0.1082  |       0.3150  |       0.1050  |       0.0200|  
|TransH         |       32.1400 |       0.1168  |       0.3350  |       0.1050  |       0.0200|  
|TransR         |       25.1200 |       0.1354  |       0.4300  |       0.1150  |       0.0200|  
|ConvKB         |       23.2650 |       0.2038  |       0.5100  |       0.2200  |       0.0750|  
|TransformerKB  |       22.6450 |       0.2446  |       0.5300  |       0.1350  |       0.0450|  

TransformerKB在MRR指标上表现最好，其余各指标稍差，分析可能是因为三元组抽取过程中存在较多错误，造成输入噪音


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





---

|       Model   |       MR      |       MRR     |       hit@10  |       hit@3   |       hit@1|   |Analogy        |       45.9100 |       0.0704  |       0.2050  |       0.0300  |       0.0050|  |ComplEx        |       50.6650 |       0.0582  |       0.1650  |       0.0100  |       0.0100|  |DistMult       |       43.8200 |       0.0876  |       0.2650  |       0.0600  |       0.0150|
|HolE           |       32.9700 |       0.1653  |       0.3200  |       0.1550  |       0.0800|  |RESCAL         |       34.0000 |       0.1173  |       0.3050  |       0.0900  |       0.0350|  |TransD         |       31.2500 |       0.1198  |       0.3400  |       0.0900  |       0.0300|  |TransE         |       54.0050 |       0.0659  |       0.1100  |       0.0450  |       0.0300|  |TransH         |       47.1550 |       0.0792  |       0.1350  |       0.0600  |       0.0300|  |TransR         |       38.0600 |       0.1016  |       0.2350  |       0.0800  |       0.0300|  |ConvKB         |       25.5650 |       0.1879  |       0.4300  |       0.2100  |       0.0800|  |TransformerKB  |       22.0300 |       0.1784  |       0.4550  |       0.1600  |       0.0650|


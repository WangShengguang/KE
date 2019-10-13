# KE
Knowledge Embedding，知识表示

主要参考OpenKE，将其中C++部分用Python重新实现，过度抽象封装的部分进行重新封装 


## 模型简介

- [知乎：Trans系列知识表示学习方法梳理](https://zhuanlan.zhihu.com/p/32993044)
- [THUNLP: Must-read papers on knowledge representation learning (KRL) / knowledge embedding (KE)](https://github.com/thunlp/KRLPapers)

---

## 1. 数据准备 
    
| input_data         | out_data       |
| ------------------ | -------------- |
| benchmarks/dataset | output/dataset |

| 数据集      | 说明                                              |
| ----------- | ------------------------------------------------- |
| lawdata     | 原始标注集通过NERE后得到                          |
| lawdata_new | 与lawdata同，避免生成数据覆盖而命名通过NERE后得到 |
| traffic_all | 未标注数据通过NERE后得到                          |


参数设置，HolE,RESCAL,TransD,TransE,TransH,TransR使用hinge loss function,margin设置为1，
其余参数见下表 

| 参数                  | value  |
| --------------------- | ------ |
| batch_size            | 16     |
| learning_rate         | 0.0001 |
| l2 regularizer lambda | 0.001  |
| dropout_keep_prob     | 0.8    |
| margin                | 1.0    |
| ent_emb_dim           | 128    |
| rel_emb_dim           | 128    |

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
| Training Set    | 263             | 98                 | 8                   |
| Development Set | 104             | 98                 | 8                   |
| Test Set        | 103             | 98                 | 8                   |

各模型在此数据集上的表现（mac_mrr）

| Model         | MR     | MRR    | hit@10 | hit@3  | hit@1  |
| ------------- | ------ | ------ | ------ | ------ | ------ |
| Analogy       | 3.5146 | 0.5669 | 0.9272 | 0.7039 | 0.3786 |
| ComplEx       | 3.4320 | 0.5897 | 0.9320 | 0.6990 | 0.4078 |
| DistMult      | 3.8641 | 0.5659 | 0.9078 | 0.6845 | 0.3932 |
| HolE          | 4.1019 | 0.5512 | 0.8981 | 0.6602 | 0.3738 |
| RESCAL        | 3.4709 | 0.5633 | 0.9320 | 0.7282 | 0.3592 |
| TransD        | 4.0340 | 0.3982 | 0.9223 | 0.6359 | 0.0922 |
| TransE        | 4.1311 | 0.3745 | 0.9223 | 0.6456 | 0.0534 |
| TransH        | 3.9660 | 0.3884 | 0.9369 | 0.6456 | 0.0728 |
| TransR        | 3.3010 | 0.5718 | 0.9563 | 0.7282 | 0.3738 |
| ConvKB        | 3.6990 | 0.5444 | 0.9272 | 0.6796 | 0.3495 |
| TransformerKB | 3.2087 | 0.6059 | 0.9563 | 0.7282 | 0.4223 |



|Analogy           & 3.515               & 0.567              & 0.927          & 35.095             & 0.121             & 0.380              \\
|ComplEx           & 3.432               & \underline{0.590}  & 0.932          & 33.125             & 0.107             & 0.340              \\
|DistMult          & 3.864               & 0.566              & 0.908          & 35.015             & 0.112             & 0.375              \\
|HolE              & 4.102               & 0.551              & 0.898          & 37.205             & 0.141             & 0.375              \\
|RESCAL            & 3.471               & 0.563              & 0.932          & 33.800             & 0.140             & 0.380              \\
|TransD            & 4.034               & 0.398              & 0.922          & 27.865             & 0.144             & 0.410              \\
|TransE            & 4.131               & 0.375              & 0.922          & 28.475             & 0.143             & 0.405              \\
|TransH            & 3.966               & 0.388              & 0.937          & 29.815             & 0.138             & 0.385              \\
|TransR            &  \underline{3.301}  & 0.572              & \textbf0.956}  & \underline{25.260} & 0.144             & 0.425              \\
|ConvKB            & 3.699              & 0.544              & 0.927          & 28.120             & \textbf{0.233}    & \textbf{0.445}     \\
|TransformerKB     & \textbf{3.209}      & \textbf{0.6060}     & \textbf{0.956} & \textbf{22.935}    & \underline{0.168} & \underline{0.440}   \\



TransformerKB在各项指标均优于其余模型


### 2.2 随机数据集
随机挑选的500篇交通判决案例通过NER&RE三元组抽取和知识融合得到新的三元组，按3:1:1划分训练集、验证集、测试集


| Datasets        | Number of Cases | Number of Entities | Number of Relations |
| --------------- | --------------- | ------------------ | ------------------- |
| Training Set    | 300             | 111                | 8                   |
| Development Set | 100             | 111                | 8                   |
| Test Set        | 100             | 111                | 8                   |

各模型在此数据集上的表现（mac_mrr）

| Model         | MR      | MRR    | hit@10 | hit@3  | hit@1  |
| ------------- | ------- | ------ | ------ | ------ | ------ |
| Analogy       | 35.0950 | 0.1212 | 0.3800 | 0.1100 | 0.0150 |
| ComplEx       | 33.1250 | 0.1066 | 0.3400 | 0.0700 | 0.0100 |
| DistMult      | 35.0150 | 0.1122 | 0.3750 | 0.1200 | 0.0050 |
| HolE          | 37.2050 | 0.1414 | 0.3750 | 0.1450 | 0.0400 |
| RESCAL        | 33.8000 | 0.1403 | 0.3800 | 0.1400 | 0.0300 |
| TransD        | 27.8650 | 0.1436 | 0.4100 | 0.1450 | 0.0300 |
| TransE        | 28.4750 | 0.1428 | 0.4050 | 0.1500 | 0.0300 |
| TransH        | 29.8150 | 0.1377 | 0.3850 | 0.1250 | 0.0300 |
| TransR        | 25.2600 | 0.1438 | 0.4250 | 0.1250 | 0.0250 |
| ConvKB        | 28.1200 | 0.2330 | 0.4450 | 0.2500 | 0.1350 |
| TransformerKB | 22.9350 | 0.1682 | 0.4400 | 0.1700 | 0.0500 |

TransformerKB在MRR指标上表现最好，其余各指标稍差，分析可能是因为三元组抽取过程中存在较多错误，造成输入噪音

We separate models into two sets, the first are ConvKB, TransformerKB, TransX (TransE, TransH, TransR, TransD) and the second are Analogy, ComplEx, DistMult, HolE, RESCAL. On the training data, the performance of TransX and ConvKB is less on MR, HRR, and Hit@10. But on the random pick data, their performance is better than the other models.  That we analysis the fluctuation is because the dataset is a little bit small, thus the random bias has affected the models. The other reason is that the training lawsuit data is much similar to the labeled data, thus intuition we should get more accurate triplets than the random pick lawsuit data. Even though, the TransX series models still have the same order on both datasets, e.g. on the performance of MR they retain a relationship of TransR > TransH > TransE > TransD. On the other hand, due to the difference in the implementation detail of the second set models, their performance order is much more variant on both datasets. As we can see the performance of the TransX series model is more stable as a better baseline. There is a model in which performance is much closer to our TransformerKB is the ConvKB model. In comparison, the performance of TransformerKB is higher than ConvKB four times and two times worse. ConvKB has once in the last, shows that its stability is less. TransformerKB has performed four times the best and twice the second, and all better than TransX series models, we can see TransformerKB has better stability and precision on this task.


---

TransformerKB does better than the closely related model ConvKB on train lawsuit data where TransformerKB gains significant improvements of 0.92 in MR, 0.08 in MRR and 0.04 in Hit@10. Except Hit@10 is a bit lower than ComlEx, our TransformerKB model has a higher score on MR and Hit@N than all other baseline model. We believe that is because the self-attention mechanism captures more connections between entities and relations.

On the random data, TransformerKB only perform the best on MR. It is possibly because the triplet extraction progress exist some invalid triplets and that cause the noise.

On both data sets, the TransX series models performed relatively well and were stable.TransE, TransH and TransR all has better MR, Hit@10 on train lawsuit data than ConvKB. We confirmed that the TranX model is a strong baseline model.

TransformerKB exceeds the TransX series models for all metrics on both datasets. Most of the time, TransformerKB is similar or better than ConvKB. It shows that TransformerKB has better performance on overall prediction and more stable than ConvKB. TransformerKB can be seen as a promotion combined with TransE and ConvKB.



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



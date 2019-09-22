# KE
Knowledge Embedding   ，知识编码

主要参考OpenKE，将其中C++部分用Python重新实现，过度抽象封装的部分进行重新封装 

---
### 实体预测
|数据集|num entity|模型|mrr|mr|hit@10|hit@3|hit@1|
|---|---|---|---|---|---|---|---|
|lawdata|98|TransE|0.3358|4.2650|0.9350|0.5950|0.0100|
|lawdata|98|ConvKB|0.5891|3.3400|0.9350|0.7250|0.3950|
|lawdata|98|TransformerKB|0.5918|3.2950|0.9400|0.7200|0.4050|
|WN18RR|40943|TransE|0.0719|8944.3050|0.1850|0.1300|0.0000|
|WN18RR|40943|ConvKB|0.1093|11796.6300|0.1800|0.1450|0.0650|
|WN18RR|40943|TransformerKB|0.0940|8300.8650|0.1600|0.1100|0.0650|
|FB15K|14951|TransE|0.1107|262.4550|0.3000|0.1250|0.0200|
|FB15K|14951|ConvKB|0.1439|205.0950|0.3100|0.1400|0.0750|
|FB15K|14951|TransformerKB|0.1069|203.9850|0.2150|0.1150|0.0450|

- OpenKE: averaged(raw): 0.156255 271.682068 0.300757 0.167426    0.084094

*model:TransformerKB FB15K, mrr:0.1069, mr:203.9850, hit_10:0.2150, hit_3:0.1150, hit_1:0.0450 


# tips
    data feed 
    多轮训练（断点续训）
    load max_step 而不是min_loss

## Reference
- [Awesome Knowledge Graph Embedding Approaches](https://gist.github.com/mommi84/07f7c044fa18aaaa7b5133230207d8d4)








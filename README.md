# KE
Knowledge  Embedding   

知识编模型




主要参考OpenKE，将其中C++部分用Python重新实现，过度抽象封装的部分进行重新封装 


### 关系识别
|数据集|num entity|模型|mrr|mr|hit@10|hit@3|hit@1|
|---|---|---|---|---|---|---|---|
|lawdata|98|TransE|0.2544|9.1359|0.7039|0.3883|0.0097|
|lawdata|98|ConvKB|0.5865|3.4466|0.9320|0.7184|0.4029|
|lawdata|98|TransformerKB|0.6082|3.3835|0.9272|0.7282|0.4320|
|WN18RR|40943|TransformerKB|0.0469|15635.0100|0.1000|0.0550|0.0250|
|WN18RR|40943|ConvKB|0.0568|12138.0100|0.0950|0.0600|0.0350|
|FB15K|14951|TransformerKB|0.0209|2380.8920|0.0405|0.0170|0.0075|

*model:WN18RR TransE, mrr:0.0035, mr:15305.3800, hit_10:0.0150, hit_3:0.0000, hit_1:0.0000
*model:FB15K TransE, mrr:0.0177, mr:5497.3300, hit_10:0.0300, hit_3:0.0100, hit_1:0.0100  


averaged(raw): 0.156255 271.682068 0.300757 0.167426    0.084094

* local train TransE
>lowdata model:TransE, mrr:0.5514, mr:3.5243, hit_10:0.9320, hit_3:0.7184, hit_1:0.3301


- [Awesome Knowledge Graph Embedding Approaches](https://gist.github.com/mommi84/07f7c044fa18aaaa7b5133230207d8d4)
 

# tips
    data feed 
    多轮训练（断点续训）

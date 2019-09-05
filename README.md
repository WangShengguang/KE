# KE
Knowledge  Embedding   

知识编模型




主要参考OpenKE，将其中C++部分用Python重新实现，过度抽象封装的部分进行重新封装 


### 关系识别
|数据集|模型|mrr|mr|hit@10|hit@3|hit@1|
|---|---|---|---|---|---|---|
|lawdata|TransE|0.2544|9.1359|0.7039|0.3883|0.0097|
|lawdata|ConvKB|0.5865|3.4466|0.9320|0.7184|0.4029|
|lawdata|TransformerKB|0.6082|3.3835|0.9272|0.7282|0.4320|

* local train 
>lowdata model:TransE, mrr:0.5514, mr:3.5243, hit_10:0.9320, hit_3:0.7184, hit_1:0.3301


- [Awesome Knowledge Graph Embedding Approaches](https://gist.github.com/mommi84/07f7c044fa18aaaa7b5133230207d8d4)
 

# tips
    data feed 
    多轮训练（断点续训）

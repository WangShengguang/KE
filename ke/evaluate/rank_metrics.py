class RankMetrics(object):
    """ 推荐系统遇上深度学习(十六)--详解推荐系统中的常用评测指标
        这些指标有的适用于二分类问题，有的适用于对推荐列表topk的评价。
            https://juejin.im/entry/5b2e690af265da595e3ccd6b
    """

    def __init__(self):
        pass

    def mrr(self, y_id, pred_ids):
        """
          Mean Reciprocal Rank
        example:
            (h,r,t) , t is unknown
                    预测之，预测结果出现的位置的倒数，越靠前越好 (1/5+1/6+1/20)/3 = 0.138888...
        :type pred_ids: list , sorted by score desc
        :type y_id: int , sorted by score desc
        :return:
        """
        return 1 / (pred_ids.index(y_id) + 1)

    def mr(self, y_id, pred_ids):
        """
            mean rank of correct entities
        example:
            (h,r,t) , t is unknown
                    预测 h or t，预测结果出现的位置，越靠前越好 (5+6+20)/3 = 10.3333...
        :return:
        """
        return pred_ids.index(y_id) + 1

    def hit_k_count(self, y_ids, pred_ids, k=1):
        """  Hit-ratio 命中率
        example:
            分母是所有的测试集合，分子是每个用户top-K推荐列表中属于测试集合的个数的总和。
            举个简单的例子，三个用户在测试集中的商品个数分别是10，12，8，
            模型得到的top-10推荐列表中，分别有6个，5个，4个在测试集中，
            那么此时HR的值是 (6+5+4)/(10+12+8) = 0.5。
        :param k:
        :return:
        """
        return sum([1 if pred in y_ids else 0 for pred in pred_ids[:k]])

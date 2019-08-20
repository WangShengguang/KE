import copy

import numpy as np


class Metrics(object):
    """
        AUC
        https://blog.csdn.net/xiaqian0917/article/details/53445071
    """

    def __init__(self):
        pass

    def macro(self):
        """
            每个sample，每个label的acc计算出来
                所有label的acc平均即为当前sample的acc
                所有sample的acc平均即为macro acc
        :return:
        """

    def micro(self):
        pass


def calcu_metrics(y_trues, y_preds, all_labels=(0, 1), mode="macro"):
    """  对于给定的测试数据集，分类器正确分类的样本数与总样本数之比。也就是损失函数是0-1损失时测试数据集上的准确率。
        多类分类问题中，分类结果一般有4种情况:
            属于类C的样本被正确分类到类C，记这一类样本数为 TP
            不属于类C的样本被错误分类到类C，记这一类样本数为 FN
            属于类别C的样本被错误分类到类C的其他类，记这一类样本数为 TN
            不属于类别C的样本被正确分类到了类别C的其他类，记这一类样本数为 FP
    :param y_trues: [[0],[1],[0]] or [[0,2],[1],[3]]
    :param y_preds: [[1],[1],[0]] or [[0,2],[1],[3]]
    :param all_labels:
                单分类：数字，回归预测值; 计算acc需要转换为二分类
                二分类 ：(0,1)
                多分类：(0,1,2,3,4,5,6)
    :return:
    """
    indicator = {k: 0 for k in ["TP", "TN", "FP", "FN"]}
    tag_dic = {tag: copy.deepcopy(indicator) for tag in all_labels}  # 针对每个tag的情况
    all_labels = set(all_labels)

    def calculate(tag_dic):
        """
        :param tag_dic: {"tag":{"TP":0,"TN":1,"FP":2,"FN":3}}
        :return:
        """
        indicators = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for tag, dic in tag_dic.items():
            TP, TN, FP, FN = dic["TP"], dic["TN"], dic["FP"], dic["FN"]
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
            indicators["accuracy"].append(accuracy)
            indicators["precision"].append(precision)
            indicators["recall"].append(recall)
            indicators["f1"].append(f1)
        # 所有label的acc平均
        accuracy = np.mean(indicators["accuracy"])
        precision = np.mean(indicators["precision"])
        recall = np.mean(indicators["recall"])
        f1 = np.mean(indicators["f1"])
        return accuracy, precision, recall, f1

    def statistics(real_pos_labels, pred_pos_labels):
        real_neg_labels = all_labels - real_pos_labels  # 真实负例
        pred_neg_labels = all_labels - pred_pos_labels  # 预测的负例
        # 指标
        true_pos_labels = pred_pos_labels & real_pos_labels  # TP
        true_neg_labels = pred_neg_labels & real_neg_labels  # TN
        false_pos_labels = pred_pos_labels - real_pos_labels  # FP
        false_neg_labels = pred_neg_labels - real_neg_labels  # FN
        # 单tag指标
        _tag_dic = {tag: {k: 0 for k in ["TP", "TN", "FP", "FN"]} for tag in all_labels}  # 针对每个tag的情况
        for tag in true_pos_labels:
            _tag_dic[tag]["TP"] += 1
            tag_dic[tag]["TP"] += 1
        for tag in true_neg_labels:
            _tag_dic[tag]["TN"] += 1
            tag_dic[tag]["TN"] += 1
        for tag in false_pos_labels:
            _tag_dic[tag]["FP"] += 1
            tag_dic[tag]["FP"] += 1
        for tag in false_neg_labels:
            _tag_dic[tag]["FN"] += 1
            tag_dic[tag]["FN"] += 1
        assert len(real_pos_labels) == len(true_pos_labels) + len(false_neg_labels)  # , ipdb.set_trace()
        return _tag_dic

    # 计算每一个label在所有sample中的micro acc
    macro = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for _y_trues, _y_preds in zip(y_trues, y_preds):
        _tag_dic = statistics(set(_y_trues), set(_y_preds))  # 一个sample
        # 宏平均每个sample都要计算
        accuracy, precision, recall, f1 = calculate(_tag_dic)  # 宏平均一个sample中每个label都计算，然后平均
        macro["accuracy"].append(accuracy)
        macro["precision"].append(precision)
        macro["recall"].append(recall)
        macro["f1"].append(f1)
    if mode == "micro":
        accuracy, precision, recall, f1 = calculate(tag_dic)
    elif mode == "macro":
        accuracy = np.mean(macro["accuracy"])
        precision = np.mean(macro["precision"])
        recall = np.mean(macro["recall"])
        f1 = np.mean(macro["f1"])
    else:
        raise ValueError(mode)
    return accuracy, precision, recall, f1

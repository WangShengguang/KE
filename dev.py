from ke.utils.logger import logging_config


def preprocess():
    """数据预处理，建立字典，提取词向量等"""
    pass


def train():
    model_name = "ConvKB"
    data_set = "lawdata"
    from ke.tf_models.train import train
    train(model_name=model_name, data_set=data_set)


def evaluate():
    model_name = "ConvKB"
    data_set = "lawdata"
    from ke.tf_models.evaluate import Prediction
    mr, mrr, hit_1, hit_3, hit_10 = Prediction(model_name, data_set).test_link_prediction()
    print("\nmrr:{:.4f}, mr:{:.4f}, hit_10:{:.4f}, hit_3:{:.4f}, hit_1:{:.4f}\n".format(mrr, mr, hit_10, hit_3, hit_1))
    accuracy, precision, recall, f1 = Prediction(model_name, data_set).test_triple_classification()
    print("\naccuracy:{:.6f}, precision:{:.6f}, recall:{:.6f}, f1:{:.6f}\n".format(accuracy, precision, recall, f1))


def main():
    logging_config("dev.log", stream_log=True, relative_path=".")
    preprocess()  # 构建词典
    # train()
    evaluate()


if __name__ == '__main__':
    # logging_config("dev.log", stream_log=True, relative_path=".")
    main()

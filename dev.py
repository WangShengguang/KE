import os

from ke.utils.gpu_selector import get_available_gpu
from ke.utils.logger import logging_config


def train():
    model_name = "ConvKB"
    data_set = "lawdata"
    # data_set = "WN18RR"
    model_name = "TransE"
    from ke.tf_models.trainer import Trainer
    Trainer(model_name=model_name, data_set=data_set).run()


def evaluate():
    model_name = "ConvKB"
    model_name = "TransE"
    data_set = "lawdata"
    from ke.tf_models.evaluator import Evaluator
    mr, mrr, hit_1, hit_3, hit_10 = Evaluator(model_name, data_set).test_link_prediction()
    print(
        "\n* mrr:{:.4f}, mr:{:.4f}, hit_10:{:.4f}, hit_3:{:.4f}, hit_1:{:.4f}\n".format(mrr, mr, hit_10, hit_3, hit_1))
    accuracy, precision, recall, f1 = Evaluator(model_name, data_set).test_triple_classification()
    print("\naccuracy:{:.6f}, precision:{:.6f}, recall:{:.6f}, f1:{:.6f}\n".format(accuracy, precision, recall, f1))


def main():
    train()
    evaluate()


if __name__ == '__main__':
    logging_config("dev.log", stream_log=True)
    available_gpu = get_available_gpu(num_gpu=1)
    print("* using GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
    main()

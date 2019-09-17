import logging
import os

from ke.utils.gpu_selector import get_available_gpu
from ke.utils.hparams import Hparams
from ke.utils.logger import logging_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def train(model_name, data_set):
    if model_name == "all" or data_set == "all":
        run_all(model_name, data_set, mode="train")
        return
    logging_config("{}-{}-train.log".format(model_name, data_set))
    from ke.trainer import Trainer
    Trainer(model_name=model_name, data_set=data_set).run()
    #
    test(model_name, data_set)


def test(model_name, data_set, log_on=False):
    if model_name == "all" or data_set == "all":
        run_all(model_name, data_set, mode="test")
        return
    if log_on:
        logging_config("{}-{}-test.log".format(model_name, data_set))
    from ke.evaluator import Evaluator
    evaluator = Evaluator(model_name, data_set, data_type="test")
    mr, mrr, hit_10, hit_3, hit_1 = evaluator.test_link_prediction()
    rank_metrics = "\n*model:{} {}, mrr:{:.4f}, mr:{:.4f}, hit_10:{:.4f}, hit_3:{:.4f}, hit_1:{:.4f}\n".format(
        model_name, data_set, mrr, mr, hit_10, hit_3, hit_1)
    print(rank_metrics)
    logging.info(rank_metrics)
    # accuracy, precision, recall, f1 = evaluator.test_triple_classification()
    # _metrics = "\nmodel:{}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}\n".format(
    #     model_name, accuracy, precision, recall, f1)
    # print(_metrics)
    # logging.info(_metrics)


def run_all(model_name, data_set, mode):
    logging_config(f"{model_name}_{data_set}_{mode}.log")
    if model_name == "all":
        all_models = models[:-1]
    else:
        all_models = [model_name]
    if data_set == "all":
        all_data_sets = data_sets[:-1]
    else:
        all_data_sets = [data_set]
    from ke.trainer import Trainer
    dataset_epoch_nums = {"WN18RR": 5, "lawdata": 100, "lawdata_new": 100, "FB15K": 3}
    for data_set in all_data_sets:
        num_epoch = dataset_epoch_nums[data_set]
        for model_name in all_models:
            Trainer(model_name=model_name, data_set=data_set, min_num_epoch=num_epoch).run()
            if mode == "train":
                test(model_name, data_set)


models = ["Analogy", "ComplEx", "DistMult", "HolE", "RESCAL",
          "TransD", "TransE", "TransH", "TransR",
          "ConvKB", "TransformerKB", "all"]

data_sets = ["lawdata", "lawdata_new", "FB15K", "WN18RR", "all"]


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # 函数名参数
    parser.add_argument('--dataset', type=str, choices=data_sets, required=True, help="数据集")
    group.add_argument('--train', type=str, choices=models, help="训练")
    group.add_argument('--test', type=str, choices=models, help="测试")
    # parse args

    args = parser.parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CPU only ...")
    else:
        available_gpu = get_available_gpu(num_gpu=1, allow_gpus=args.allow_gpus)  # default allow_gpus 0,1,2,3
        os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
        print("* using GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    # set_process_name(args.process_name)  # 设置进程名
    if args.train:
        train(model_name=args.train, data_set=args.dataset)
    elif args.test:
        test(model_name=args.test, data_set=args.dataset, log_on=True)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage.py  --train ConvKB --dataset lawdata  
        nohup python3 manage.py --train TransformerKB --dataset WN18RR --process_name TW &
        nohup python3 manage.py --test TransformerKB --dataset FB15K --process_name TF &\
        python manage.py  --train all --dataset lawdata_new  
    """
    main()

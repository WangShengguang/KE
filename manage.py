import logging
import os
from collections import defaultdict

from ke.utils.gpu_selector import get_available_gpu
from ke.utils.hparams import Hparams
from ke.utils.logger import logging_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def test(model_name, data_set, _tqdm=True):
    from ke.evaluator import Evaluator
    evaluator = Evaluator(model_name, data_set, load_model=True)
    mr, mrr, hit_10, hit_3, hit_1 = evaluator.test_link_prediction(data_type="test", _tqdm=_tqdm)
    rank_metrics = "\n*model:{}, {}, mrr:{:.3f}, mr:{:.3f}, hit_10:{:.3f}, hit_3:{:.3f}, hit_1:{:.3f}\n".format(
        model_name, data_set, mrr, mr, hit_10, hit_3, hit_1)
    print(rank_metrics)
    logging.info(rank_metrics)
    # accuracy, precision, recall, f1 = evaluator.test_triple_classification()
    # _metrics = "\nmodel:{}, accuracy:{:.3f}, precision:{:.3f}, recall:{:.3f}, f1:{:.3f}\n".format(
    #     model_name, accuracy, precision, recall, f1)
    # print(_metrics)
    # logging.info(_metrics)
    return mr, mrr, hit_10, hit_3, hit_1


def run_all(model_name, data_set, mode):
    logging_config(f"{model_name}_{data_set}_{mode}.log")
    all_models = models if model_name == "all" else [model_name]
    all_data_sets = data_sets if data_set == "all" else [data_set]
    dataset2epoch_nums = {"lawdata": 1000, "lawdata_new": 1000,
                          "traffic": 10, "traffic_all": 10, "traffic_500": 1000,
                          "FB15K": 3, "WN18RR": 5}
    all_rank_metrics = defaultdict(list)
    for data_set in all_data_sets:
        num_epoch = dataset2epoch_nums[data_set]
        for model_name in all_models:
            if mode == "train":
                # if model_name in ["TransformerKB", "TransR"]:
                #     num_epoch *= 4
                from ke.trainer import Trainer
                Trainer(model_name=model_name, data_set=data_set, min_num_epoch=num_epoch).run()
            # 每个模型 训练结束测试
            rank_metrics = test(model_name, data_set)
            all_rank_metrics[data_set].append([model_name] + list(rank_metrics))
    for dataset, _rank_metrics in all_rank_metrics.items():
        print()
        title = "\t|\t".join(["\tModel", "MR", "MRR", "hit@10", "hit@3", "hit@1"])
        title = "|" + title + "|"
        lines = []
        for _rank in _rank_metrics:
            _model_name = _rank[0] + '\t' if len(_rank[0]) < 7 else _rank[0]
            _metrics = [f"{v:.3f}" for v in _rank[1:]]
            line = "\t|\t".join([_model_name] + _metrics)
            line = "|" + line + "|"
            lines.append(line)
        print("\n----{}------\n".format(dataset))
        print("\n".join([title] + lines))
        print()


def export_embedding(data_set):
    logging_config(f"export_{data_set}_embedding.log")
    from ke.export_embedding import EmbeddingExporter
    for model_name in models:
        embedding_save_path = EmbeddingExporter(data_set=data_set, model_name=model_name).export_embedding()
        print("embedding save to : {}".format(embedding_save_path))


def sim_rank(data_set):
    logging_config(f"sim_rank_{data_set}.log")
    from triple_sim_rank.sim_rank import create_all_sim_rank
    create_all_sim_rank(data_set, models)


models = ["Analogy", "ComplEx", "DistMult", "HolE", "RESCAL",
          "TransD", "TransE", "TransH", "TransR",
          "ConvKB", "TransformerKB"]

data_sets = [
    "traffic_500",
    "lawdata",
    "lawdata_new",
    # "traffic", "traffic_all",
    # "traffic_500",
    "FB15K", "WN18RR"
]


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # 函数名参数
    parser.add_argument('--dataset', type=str, choices=data_sets + ["all"], required=True, help="数据集")
    group.add_argument('--train', type=str, choices=models + ["all"], help="训练")
    group.add_argument('--test', type=str, choices=models + ["all"], help="测试")
    group.add_argument('--export_embedding', action="store_true", help="导出entity embedding & relation embedding")
    group.add_argument('--sim_rank', action="store_true", help="推荐相似案由")
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
        run_all(model_name=args.train, data_set=args.dataset, mode="train")
    elif args.test:  # args.test:
        run_all(model_name=args.test, data_set=args.dataset, mode="test")
    elif args.export_embedding:
        export_embedding(data_set=args.dataset)
    elif args.sim_rank:
        sim_rank(data_set=args.dataset)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python3 manage.py  --train ConvKB --dataset lawdata  
        python3 manage.py --train TransformerKB --dataset WN18RR --process_name TW &
        python3 manage.py --test TransformerKB --dataset FB15K --process_name TF &
        python3 manage.py --export_embedding
        python3 manage.py -m ipdb --sim_rank --dataset traffic
        nohup python3 manage.py  --train all --dataset lawdata_new   &>train_all.out&
    """
    main()

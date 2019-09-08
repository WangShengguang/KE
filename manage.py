import logging
import os

from ke.utils.gpu_selector import get_available_gpu
from ke.utils.hparams import Hparams, set_process_name
from ke.utils.logger import logging_config


def train(model_name, data_set):
    logging_config("{}-{}-train.log".format(model_name, data_set))
    from ke.tf_models.trainer import Trainer
    Trainer(model_name=model_name, data_set=data_set).run()


def test(model_name, data_set):
    logging_config("{}-{}-test.log".format(model_name, data_set))
    from ke.tf_models.evaluator import Evaluator
    evaluator = Evaluator(model_name, data_set, data_type="test")
    mr, mrr, hit_10, hit_3, hit_1 = evaluator.test_link_prediction()
    rank_metrics = "\n*model:{}, mrr:{:.4f}, mr:{:.4f}, hit_10:{:.4f}, hit_3:{:.4f}, hit_1:{:.4f}\n".format(
        model_name, mrr, mr, hit_10, hit_3, hit_1)
    print(rank_metrics)
    logging.info(rank_metrics)
    # accuracy, precision, recall, f1 = evaluator.test_triple_classification()
    # _metrics = "\nmodel:{}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}\n".format(
    #     model_name, accuracy, precision, recall, f1)
    # print(_metrics)
    # logging.info(_metrics)


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # 函数名参数
    parser.add_argument('--dataset', type=str,
                        choices=["lawdata", "FB15K", "WN18RR"],
                        required=True,
                        help="数据集")
    models = ["ConvKB", "TransE", "TransformerKB"]
    group.add_argument('--train', type=str,
                       choices=models,
                       help="训练")
    group.add_argument('--test', type=str,
                       choices=models,
                       help="测试")
    # parse args
    args = parser.parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CPU only ...")
    else:
        available_gpu = get_available_gpu(num_gpu=1, allow_gpus=args.allow_gpus)  # default allow_gpus 0,1,2,3
        os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
        print("* using GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    set_process_name(args.process_name)  # 设置进程名
    #
    if args.train:
        train(model_name=args.train, data_set=args.dataset)
    elif args.test:
        test(model_name=args.test, data_set=args.dataset)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage.py  --train ConvKB --dataset lawdata  
        nohup python3 manage.py --train TransformerKB --dataset WN18RR --process_name TW &
        nohup python3 manage.py --test TransformerKB --dataset FB15K --process_name TF &
        
    """
    main()

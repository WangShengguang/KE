import os

from ke.utils.gpu_selector import get_available_gpu
from ke.utils.hparams import Hparams
from ke.utils.logger import logging_config


def train(model_name, data_set):
    logging_config("{}-{}.log".format(model_name, data_set))
    from ke.tf_models.trainer import Trainer
    Trainer(model_name=model_name, data_set=data_set).run()


def test(model_name, data_set):
    logging_config("{}-{}.log".format(model_name, data_set))
    from ke.tf_models.evaluator import Evaluator
    evaluator = Evaluator(model_name, data_set, data_type="test")
    mr, mrr, hit_1, hit_3, hit_10 = evaluator.test_link_prediction()
    print(
        "\n* mrr:{:.4f}, mr:{:.4f}, hit_10:{:.4f}, hit_3:{:.4f}, hit_1:{:.4f}\n".format(mrr, mr, hit_10, hit_3, hit_1))
    accuracy, precision, recall, f1 = evaluator.test_triple_classification()
    print("\naccuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}\n".format(accuracy, precision, recall, f1))


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # 函数名参数
    parser.add_argument('--dataset', type=str, default="lawdata",
                        choices=["lawdata", "FB15K", "WN18RR"],
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
    available_gpu = get_available_gpu(num_gpu=1, allow_gpus=args.allow_gpus)  # default allow_gpus 0,1,2,3
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
    print("* using GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    #
    if args.train:
        train(model_name=args.train, data_set=args.dataset)
    elif args.test:
        test(model_name=args.test, data_set=args.dataset)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage.py  --train ConvKB --dataset lawdata  
    """
    main()

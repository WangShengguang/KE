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
    from ke.tf_models.trainer import Trainer
    Trainer(model_name=model_name, data_set=data_set).run()


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # 函数名参数
    parser.add_argument('--dataset', type=str, default="lawdata",
                        choices=["BERTCRF", "BERTSoftmax"],  # model name
                        help="Named Entity Recognition，实体识别")
    models = ["ConvKB", "TransE"]
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
        test(model_name=args.train, data_set=args.dataset)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage.py  --train ConvKB --dataset lawdata  
    """
    main()

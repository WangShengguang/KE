import os
from functools import partial

from ke.utils.hparams import ArgCallback, Hparams
from ke.utils.logger import logging_config


def preprocess():
    """数据预处理，建立字典，提取词向量等"""
    pass


def train(model_name):
    """模型训练"""
    if model_name == "bilstm_crf":
        pass


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --level
        --gpu
    '''
    parser = Hparams().parser
    # 函数名参数
    parser.add_argument('--preprocess', action="store_true", help="数据预处理")
    parser.add_argument('--train', type=str,
                        choices=["ConvKB", "keras_bilstm_crf"],
                        help="模型训练")
    parser.add_argument('--test', action="store_true", help="测试")
    # parse args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 指定GPU 0 or 0,1,2 ...
    logger = partial(logging_config, relative_path=args.relative_path, stream_log=args.stream_log)
    #
    if args.train:
        model_name = args.train
        logging_config("{}.log".format(model_name))
        train(model_name)
    else:
        # 不需要接收参数的函数可放在此处执行;日志：函数名.log
        ArgCallback(vars(args), __name__)  # 自动寻找并调用此模块(manage.py) 中与参数名同名的函数


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage.py --preprocess #执行preprocess()进行数据预处理
        python manage.py --train bilstm_crf  # train("bilstm_crf") 
        python manage.py --eval bilstm_crf  # eval("bilstm_crf") 

        nohup python manage.py --preprocess --gpu 0  &>preprocess.nohup.out&
    """

    main()

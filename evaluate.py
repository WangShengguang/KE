import os
from functools import partial

from ke.utils.hparams import ArgCallback, Hparams
from ke.utils.logger import logging_config


def evaluate(model_name):
    """
    :param model_name:
    :return:
    """
    pass


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --level
        --gpu
    '''
    parser = Hparams().parser
    # 函数名参数   # 函数名参数
    parser.add_argument('--eval_siamese', action="store_true", help="孪生网络")
    parser.add_argument('--evaluate', type=str,
                        choices=[
                            "knn",
                            "siamese",
                            "text_cnn",
                            "keras_text_cnn", "keras_cnn_rnn", "keras_cnn_pair_rnn", "keras_bilstm",
                            "keras_rnn_attention", "keras_capsule_gru", ],
                        help="评价函数")
    # parse args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 指定GPU 0 or 0,1,2 ...
    logger = partial(logging_config, relative_path=args.relative_path, stream_log=args.stream_log)

    if args.evaluate:
        logger("evaluate_{}.log".format(args.evaluate))
        evaluate(args.evaluate)
    else:
        # ArgCallback(vars(args), model_name="evaluate")  # 调用参数名对应的函数。日志：函数名.log
        ArgCallback(vars(args), model_name=__name__)  # 调用参数名对应的函数。日志：函数名.log


if __name__ == '__main__':
    """ 此文件只调用测试模型的函数，训练转到manage.py文件执行
    examples:
        python3 evaluate.py --cnn_predict # 调用cnn_predict()函数
        python3 evaluate.py --evaluate keras_text_cnn  #evaluate()函数
        python3 evaluate.py --evaluate knn  --gpu 0  &>knn.out&
    """

    main()

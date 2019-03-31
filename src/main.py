# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import pandas as pd
import functools
import DataSet
import Estimator
import os
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument("--n_epoch", default=5, type=int)
parser.add_argument("--data_path", default="./data/sample-1m", type=str)
parser.add_argument("--emb_size", default=64, type=int)
parser.add_argument("--cuda", default='1', type=str)


def main(argv):
    args = parser.parse_args(argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(
            gpu_options={"allow_growth": True},
        ))

    '''
    自定义DNN
    model_fn是模型定义
    '''
    classifier = tf.estimator.Estimator(
        model_fn=Estimator.my_model,
        model_dir="./model/{}".format(int(time.time())),
        config=run_config,
        params={
            "emb_size": args.emb_size,
        },
    )

    '''
    estimator提供train, evaluate, predict方法，
    train方法需要输入无参方法提供数据
    '''
    train_input_fn = functools.partial(
        DataSet.input_fn,
        data_path="{}.train".format(args.data_path),
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
    )

    classifier.train(
        input_fn=train_input_fn,
    )

    # Evaluate the model.
    eval_input_fn = functools.partial(
        DataSet.input_fn,
        data_path="{}.test".format(args.data_path),
        batch_size=args.batch_size,
    )
    eval_result = classifier.evaluate(
        input_fn=eval_input_fn,
        steps=1000,
    )
    print(eval_result)
    for k, v in eval_result.items():
        print('Test set {}: {}'.format(k, v))

    '''
    导出estimator为pb模型文件，需要提供serving_input_receiver_fn方法
    '''
    # TensorFolw standard serving input, for tensorflow serving
    feature_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    print('Exported to {}'.format(export_dir))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    # main(sys.argv)

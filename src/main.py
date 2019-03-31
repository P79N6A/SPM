# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import functools
import DataSet
import Estimator
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--train_steps', default=1000, type=int)
parser.add_argument("--n_epoch", default=5, type=int)
parser.add_argument("--data_path", default="./data/sample-1m", type=str)
parser.add_argument("--emb_size", default=64, type=int)
parser.add_argument("--conv_size", default=5, type=int)
parser.add_argument("--n_attention", default=3, type=int)
parser.add_argument("--dropout", default=0.5, type=int)
parser.add_argument("--l2", default=1e-4, type=float)


def main(argv):
    args = parser.parse_args(argv[1:])

    n_word = 100000
    n_id = 100000

    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(
            gpu_options={"allow_growth": True},
        ))

    '''
    自定义DNN
    model_fn是模型定义
    '''
    classifier = tf.estimator.Estimator(
        model_fn=Estimator.model_fn,
        model_dir="./model/{}".format(int(time.time())),
        config=run_config,
        params={
            "emb_size": args.emb_size,
            "n_word": n_word,
            "n_id": n_id,
            "conv_size": args.conv_size,
            "n_attention": args.n_attention,
            "dropout": args.drop_out,
            "l2": args.l2,
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
    feature_spec = {
        "feat_ids": tf.FixedLenFeature(dtype=tf.int64, shape=[None, args.field_size]),
        "feat_vals": tf.FixedLenFeature(dtype=tf.float32, shape=[None, args.field_size])
    }
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    export_dir = classifier.export_savedmodel('export', serving_input_receiver_fn)
    print('Exported to {}'.format(export_dir))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    # main(sys.argv)

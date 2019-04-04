# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import estimator.MaxAttention as estimator
import time
import json

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--train_steps', default=1000, type=int)
parser.add_argument("--n_epoch", default=10, type=int)
parser.add_argument("--data_path", default="./data/multi_data7/multi_data7", type=str)
parser.add_argument("--emb_size", default=64, type=int)
parser.add_argument("--conv_size", default=5, type=int)
parser.add_argument("--n_attention", default=3, type=int)
parser.add_argument("--dropout", default=0.3, type=int)
parser.add_argument("--l2", default=1e-4, type=float)


def main(argv):
    args = parser.parse_args(argv[1:])

    with open("{}.param.json".format(args.data_path)) as f:
        params = json.load(f)

    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(
            gpu_options={"allow_growth": True},
        ))

    '''
    自定义DNN
    model_fn是模型定义
    '''
    classifier = tf.estimator.Estimator(
        model_fn=estimator.model_fn,
        model_dir="./model/{}".format(int(time.time())),
        config=run_config,
        params={
            "emb_size": args.emb_size,
            "n_word": params["n_word"],
            "n_id": params["n_id"],
            "conv_size": args.conv_size,
            "n_attention": args.n_attention,
            "dropout": args.dropout,
            "l2": args.l2,
        },
    )

    '''
    estimator提供train, evaluate, predict方法，
    train方法需要输入无参方法提供数据
    '''
    for epoch in range(1, args.n_epoch + 1):
        train_input_fn = lambda: estimator.input_fn(
            filenames="{}.train2.tfrecord".format(args.data_path),
            batch_size=args.batch_size,
        )

        classifier.train(
            input_fn=train_input_fn,
            # steps=100,
        )

        # Evaluate the model.
        eval_input_fn = lambda: estimator.input_fn(
            filenames="{}.test2.tfrecord".format(args.data_path),
            batch_size=args.batch_size,
        )

        eval_result = classifier.evaluate(
            input_fn=eval_input_fn,
            steps=1000,
        )
        # print(eval_result)
        print("Epoch: {}".format(epoch))
        for k, v in eval_result.items():
            print('Test set {}: {}'.format(k, v))

    '''
    导出estimator为pb模型文件，需要提供serving_input_receiver_fn方法
    '''
    # TensorFolw standard serving input, for tensorflow serving
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(estimator.FEATURES)
    export_dir = classifier.export_savedmodel('export', serving_input_receiver_fn)
    print('Exported to {}'.format(export_dir))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    # main(sys.argv)

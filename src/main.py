#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
from estimator import \
    MaxAttention, PNN, DynamicAttention, NCF, \
    MultiHeadAttention, MultiViewAttention, DynamicAttentionWithMF
from estimator.common import input_fn
from utils import report
import time
import json
import shutil
import os

parser = argparse.ArgumentParser()
# data
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--train_steps', default=1000, type=int)
parser.add_argument("--n_epoch", default=30, type=int)
parser.add_argument("--data_path", default="./data/multi_data7_300k_pre/multi_data7_300k_pre", type=str)
# common params
parser.add_argument("--emb_size", default=64, type=int)
parser.add_argument("--conv_size", default=5, type=int)
parser.add_argument("--dropout", default=0.3, type=int)
parser.add_argument("--l2", default=1e-4, type=float)
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--logdir", default="", type=str)
parser.add_argument("--shuffle_size", default=100, type=int)
parser.add_argument("--load_pre", default=0, type=int)
parser.add_argument("--load_wcls", default=0, type=int)
# model select
parser.add_argument("--model", default="NCF", type=str)
# SPM
parser.add_argument("--n_attention", default=5, type=int)
parser.add_argument("--cross", default=1, type=int)
parser.add_argument("--mf", default=0, type=int)


def load_w2v(fp_pre):
    data = []
    with open(fp_pre, encoding="gbk") as f:
        for line in f:
            line = line.strip().split()
            if len(line) != 200:
                line = line[1:]
            line = [float(x) for x in line]
            data.append(line)
    return data


def main(argv):
    args = parser.parse_args(argv[1:])
    print(args)

    with open("{}.param.json".format(args.data_path)) as f:
        params = json.load(f)

    logdir = args.logdir
    if not logdir:
        logdir = "./model/{}".format(int(time.time()))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    w2v_pre = 0
    if args.load_pre:
        fp_pre = "{0}.w2v".format(args.data_path)
        w2v_pre = load_w2v(fp_pre)

    w_cls = 0
    if args.load_wcls:
        fp_cls = "{0}.word_cls".format(args.data_path)
        with open(fp_cls) as f:
            w_cls = f.read().strip().split("\n")
            w_cls = [int(line.split()[(args.n_attention - 1) // 2]) for line in w_cls]

    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(
            gpu_options={"allow_growth": True},
        ))

    '''
    自定义DNN
    model_fn是模型定义
    '''
    models = {
        "SPM": MaxAttention,
        "PNN": PNN,
        "DMA": DynamicAttention,
        "MA": MultiHeadAttention,
        "MVA": MultiViewAttention,
        "DMAMF": DynamicAttentionWithMF,
        "NCF": NCF,
    }
    my_estimator = models[args.model]

    classifier = tf.estimator.Estimator(
        model_fn=my_estimator.model_fn,
        model_dir=logdir,
        config=run_config,
        params={
            "emb_size": args.emb_size,
            "n_word": params["n_word"],
            "n_job": params["n_job"],
            "n_person": params["n_person"],
            "conv_size": args.conv_size,
            "n_attention": args.n_attention,
            "dropout": args.dropout,
            "l2": args.l2,
            "lr": args.lr,
            "w2v_pre": w2v_pre,
            "w_cls": w_cls,
            "cross": args.cross,
            "mf": args.mf,
        },
    )

    report.reportmetrics(
        [{"sMetricsName": "auc", "sMetricsValue": 0.5}]
    )
    for epoch in range(1, args.n_epoch + 1):
        train_input_fn = lambda: input_fn(
            filenames="{}.train2.tfrecord".format(args.data_path),
            batch_size=args.batch_size,
            shuffle=args.batch_size,
        )

        classifier.train(
            input_fn=train_input_fn,
            # steps=100,
        )

        # Evaluate the model.
        eval_input_fn = lambda: input_fn(
            filenames="{}.test2.tfrecord".format(args.data_path),
            batch_size=args.batch_size,
            shuffle=args.shuffle_size,
        )

        eval_result = classifier.evaluate(
            input_fn=eval_input_fn,
            steps=1000,
        )
        rpt = [{"sMetricsName": "epoch", "sMetricsValue": epoch}]
        print("epoch: {}".format(epoch))
        for k, v in eval_result.items():
            print('Test set {}: {}'.format(k, v))
            rpt.append({"sMetricsName": k, "sMetricsValue": float(v)})
        report.reportmetrics(rpt)

    '''
    导出estimator为pb模型文件，需要提供serving_input_receiver_fn方法
    '''
    # TensorFolw standard serving input, for tensorflow serving
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(my_estimator.FEATURES)
    export_dir = classifier.export_savedmodel('export', serving_input_receiver_fn)
    print('Exported to {}'.format(export_dir))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    # main(sys.argv)

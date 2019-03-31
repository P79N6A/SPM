# -*- coding: utf-8 -*-
import tensorflow as tf


def feature_emb()

def my_model(features, labels, mode, params):
    '''
    :param features: dict of tf.Tensor
    :param labels: tf.Tensor
    :param mode: tf.estimator.ModeKeys
    :param params: customer params
    :return:
    '''
    n_word = params["n_word"]
    emb_size = params["emb_size"]

    '''
    features是包含一个dict，key为特征名，value为原始特征Tensor
    '''

    jids = features["jids"]
    jds = features["jds"]
    pids = features["pids"]
    cvs = features["cvs"]


    word_emb = tf.Variable(
        initial_value=tf.random_normal(shape=(n_word, emb_size)),
        name = "word_emb",
    )
    jds = feature_emb(jds, emb_size, name="jd_emb")
    cvs = feature_emb(cvs, )    

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, units=1)
    probs = tf.nn.sigmoid(logits, name="output")

    '''
    model_fn的预期返回值是tf.estimator.EstimatorSpec对象
    调用estimator对象的不同方法时，会传入相应的mode参数
    mode取值范围：{
        tf.estimator.ModeKeys.PREDICT,
        tf.estimator.ModeKeys.EVAL,
        tf.estimator.ModeKeys.TRAIN,
    }
    需要对三个mode指定返回值
    '''
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': probs,
            'logits': logits,
        }
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:\
            tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs,
        )

    loss = tf.losses.log_loss(
        labels=labels,
        predictions=tf.squeeze(probs),
    )
    auc = tf.metrics.auc(
        labels=labels,
        predictions=tf.squeeze(probs),
        name="auc_op",
    )
    metrics = {
        "auc": auc,
    }
    '''
    调用tf.summary可以记录关心的变量，数据会写入tensorboard
    scalar记录单个值
    histogram记录一个张量的值分布
    '''
    tf.summary.scalar("auc", auc[0])

    '''
    指定estimator.evaluate的操作
    '''
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    '''
    指定estimator.train的操作
    '''
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# -*- coding: utf-8 -*-
import tensorflow as tf


FEATURES = {
    "jids": tf.FixedLenFeature([], tf.int64),
    "jds": tf.FixedLenFeature([500], tf.int64),
    "jd_lens": tf.FixedLenFeature([], tf.int64),
    "pids": tf.FixedLenFeature([], tf.int64),
    "cvs": tf.FixedLenFeature([500], tf.int64),
    "cv_lens": tf.FixedLenFeature([], tf.int64),
}


def model_fn(features, labels, mode, params):
    '''
    :param features: dict of tf.Tensor
    :param labels: tf.Tensor
    :param mode: tf.estimator.ModeKeys
    :param params: customer params
    :return:
    '''
    n_job = params["n_job"]
    n_person = params["n_person"]
    emb_size = params["emb_size"]
    l2 = params["l2"]
    lr = params["lr"]

    '''
    features是包含一个dict，key为特征名，value为原始特征Tensor
    '''

    jids = features["jids"]
    pids = features["pids"]

    with tf.variable_scope("user_idx"):
        job_emb = tf.Variable(
            initial_value=tf.random_normal(
                shape=(n_job, emb_size),
                stddev=1 / n_job ** (1 / 2),
            ),
            name="job_emb",
        )
        j_emb = tf.nn.embedding_lookup(job_emb, jids)

        person_emb = tf.Variable(
            initial_value=tf.random_normal(
                shape=(n_person, emb_size),
                stddev=1 / n_person ** (1 / 2),
            ),
            name="person_emb",
        )        
        p_emb = tf.nn.embedding_lookup(person_emb, pids)

    logits = tf.reduce_sum(
        tf.multiply(j_emb, p_emb),
        axis=-1
    )
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

    with tf.variable_scope("loss"):
        fit_label = labels[:, 2]
        loss = tf.losses.log_loss(
            labels=fit_label,
            predictions=tf.squeeze(probs),
        )

        if l2:
            l2_params = [
                j_emb,
                p_emb
            ]
            l2_loss = sum([tf.nn.l2_loss(x) for x in l2_params])
            loss += l2_loss * l2

        auc = tf.metrics.auc(
            labels=fit_label,
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

    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


if __name__ == "__main__":
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="test",
        params={
            "emb_size": 64,
            "n_word": 1000,
            "n_job": 1000,
            "n_person": 1000,
            "conv_size": 3,
            "n_attention": 1,
            "dropout": 0.3,
            "l2": 0,
            "lr": 0.001,
        },
    )
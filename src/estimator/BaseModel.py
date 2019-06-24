# -*- coding: utf-8 -*-
import tensorflow as tf
import multiprocessing

FEATURES = {
    "jids": tf.FixedLenFeature([], tf.int64),
    "jds": tf.FixedLenFeature([500], tf.int64),
    "jd_lens": tf.FixedLenFeature([], tf.int64),
    "pids": tf.FixedLenFeature([], tf.int64),
    "cvs": tf.FixedLenFeature([500], tf.int64),
    "cv_lens": tf.FixedLenFeature([], tf.int64),
}


def cnn(x, conv_size):
    x = tf.expand_dims(x, axis=-1)
    x = tf.layers.conv2d(
        inputs=x,
        filters=1,
        kernel_size=[conv_size, 1],
        strides=[1, 1],
        padding='same',
        activation=tf.nn.relu,
    )
    x = tf.squeeze(x, axis=3)
    return x


def mlp(features, emb_dim, dropout, training):
    features = tf.layers.batch_normalization(features)
    if dropout:
        features = tf.layers.dropout(
            inputs=features,
            rate=dropout,
            training=training,
        )
    features = tf.layers.dense(
        features,
        units=emb_dim,
        activation=tf.nn.relu,
    )
    features = tf.layers.batch_normalization(features)
    predict = tf.layers.dense(
        features,
        units=1,
    )
    return predict


def model_fn(features, labels, mode, params):
    '''
    :param features: dict of tf.Tensor
    :param labels: tf.Tensor
    :param mode: tf.estimator.ModeKeys
    :param params: customer params
    :return:
    '''
    n_word = params["n_word"]
    n_job = params["n_job"]
    n_person = params["n_person"]
    emb_size = params["emb_size"]
    conv_size = params["conv_size"]
    dropout = params["dropout"]
    l2 = params["l2"]
    lr = params["lr"]
    w2v_pre = params["w2v_pre"]
    mf = params["mf"]

    '''
    features是包含一个dict，key为特征名，value为原始特征Tensor
    '''

    jids = features["jids"]
    jds = features["jds"]
    pids = features["pids"]
    cvs = features["cvs"]

    with tf.variable_scope("CNN"):
        if w2v_pre:
            word_emb = tf.Variable(initial_value=w2v_pre, name="word_emb")
        else:
            word_emb = tf.Variable(
                initial_value=tf.random_normal(
                    shape=(n_word, emb_size),
                    stddev=1 / n_word ** (1 / 2),
                ),
                name="word_emb",
            )

        jds_emb = tf.nn.embedding_lookup(word_emb, jds)
        cvs_emb = tf.nn.embedding_lookup(word_emb, cvs)

        if conv_size:
            jds_emb = cnn(jds_emb, conv_size)
            cvs_emb = cnn(cvs_emb, conv_size)

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

    with tf.variable_scope("pooling"):
        jd_global_vecs = tf.reduce_max(jds_emb, axis=1)
        cv_global_vecs = tf.reduce_max(cvs_emb, axis=1)

    features = tf.concat(
        values=[j_emb, jd_global_vecs, p_emb, cv_global_vecs],
        axis=-1,
    )

    with tf.variable_scope("mlp"):
        logits = mlp(
            features=features,
            emb_dim=emb_size,
            dropout=dropout,
            training=(mode == tf.estimator.ModeKeys.TRAIN)
        )

    if mf:
        mf_logits = tf.reduce_sum(
            tf.multiply(j_emb, p_emb),
            axis=-1,
            keepdims=True,
        )
        # logits += mf_logits
        logits = tf.add(logits, mf_logits)

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
                jds_emb,
                cvs_emb,
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
    tf.enable_eager_execution()

# -*- coding: utf-8 -*-
import tensorflow as tf
import multiprocessing


def multi_head_attention(queries: tf.Tensor, keys: tf.Tensor, keys_length):
    """
    :param queries: [B, 3, 64]
    :param keys: [B, 500, 64]
    :param keys_length: [B]
    :return: 2d array
    """

    d = queries.shape.as_list()[-1]
    scores = tf.divide(
        tf.matmul(queries, keys, transpose_b=True),
        d ** (1 / 2),
    )
    scores = tf.reduce_max(scores, axis=-2, keepdims=True)

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, 500]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, 500]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, 500]

    # Activation
    weight = tf.nn.softmax(scores)  # [B, 1, 500]

    weighted_features = tf.matmul(weight, keys)  # [B, 1, 64]
    weighted_features = tf.squeeze(weighted_features, axis=1) # [B, 64]

    queries = tf.reduce_mean(queries, axis=-2)
    return weight, weighted_features, queries


def cnn(x: tf.Tensor, conv_size):
    x = tf.expand_dims(x, axis=-1)
    x = tf.layers.conv2d(
        inputs=x,
        filters=1,
        kernel_size=[conv_size, 1],
        strides=[1, 1],
        padding='same',
        activation=tf.nn.relu,
        name='cnn',
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
    n_id = params["n_id"]
    emb_size = params["emb_size"]
    conv_size = params["conv_size"]
    n_attention = params["n_attention"]
    dropout = params["dropout"]
    l2 = params["l2"]

    '''
    features是包含一个dict，key为特征名，value为原始特征Tensor
    '''

    jids = features["jids"]
    jds = features["jds"]
    jd_lens = features["jd_lens"]
    pids = features["pids"]
    cvs = features["cvs"]
    cv_lens = features["cv_lens"]

    with tf.variable_scope("CNN"):
        word_emb = tf.Variable(
            initial_value=tf.random_normal(
                shape=(n_word, emb_size),
                stddev=1 / n_word ** (1 / 2),
            ),
            name="word_emb",
        )
        jds = tf.nn.embedding_lookup(word_emb, jds)
        jds = cnn(jds, conv_size)
        cvs = tf.nn.embedding_lookup(word_emb, cvs)
        cvs = cnn(cvs, conv_size)

    with tf.variable_scope("user_idx"):
        id_emb = tf.Variable(
            initial_value=tf.random_normal(
                shape=(n_id, n_attention, emb_size),
                stddev=1 / n_word ** (1 / 2),
            ),
            name="id_emb",
        )
        jids = tf.nn.embedding_lookup(id_emb, jds)
        pids = tf.nn.embedding_lookup(id_emb, cvs)

    with tf.variable_scope("attention"):
        jd_weights, jd_weighted_vecs, pids = multi_head_attention(pids, jds, jd_lens)
        cv_weights, cv_weighted_vecs, jids = multi_head_attention(jids, cvs, cv_lens)

    with tf.variable_scope("pooling"):
        jd_global_vecs = tf.reduce_max(jds)
        cv_global_vecs = tf.reduce_max(cvs)

    features = tf.concat([
        jids, jd_global_vecs, jd_weighted_vecs, pids, cv_global_vecs, cv_weighted_vecs
    ])

    logits = mlp(
        features=features,
        emb_dim=emb_size,
        dropout=dropout,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
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

    loss = tf.losses.log_loss(
        labels=labels,
        predictions=tf.squeeze(probs),
    )
    l2_loss = sum([tf.nn.l2_loss(x) for x in tf.trainable_variables()])
    loss += l2_loss * l2

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


def input_fn(filenames, batch_size=32, shuffle=False):
    print("parsing", filenames)
    def _parse_fn(record):
        features = {
            "labels": tf.FixedLenFeature([3], tf.int32),
            "jids": tf.FixedLenFeature([], tf.int32),
            "jds": tf.FixedLenFeature([500], tf.int32),
            "jd_lens": tf.FixedLenFeature([], tf.int32),
            "pids": tf.FixedLenFeature([], tf.int32),
            "cvs": tf.FixedLenFeature([500], tf.int32),
            "cv_lens": tf.FixedLenFeature([], tf.int32),
        }
        parsed = tf.parse_single_example(record, features)
        labels = parsed.pop("labels")
        return parsed, labels
    dataset = tf.data.TFRecordDataset(filenames).map(
        _parse_fn, num_parallel_calls=multiprocessing.cpu_count()
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.batch(batch_size)
    itetator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = itetator.get_next()
    return batch_features, batch_labels


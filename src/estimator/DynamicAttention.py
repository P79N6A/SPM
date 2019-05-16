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


def dynamic_attention(queries: tf.Tensor, keys: tf.Tensor, keys_length):
    """
    :param queries: [B, 3, 64]
    :param keys: [B, 500, 64]
    :param keys_length: [B]
    :return: 2d array
    """

    d = queries.shape.as_list()[-1]
    n_query = queries.shape.as_list()[-2]

    scores = tf.divide(
        tf.matmul(queries, keys, transpose_b=True),
        d ** (1 / 2),
    )
    # scores = tf.reduce_max(scores, axis=-2, keepdims=True)
    related_queries = tf.math.argmax(scores, axis=-2)
    zeros = tf.zeros_like(keys)

    related_features = list()
    for i in range(n_query):
        related_feature = tf.map_fn(
            tf.where(condition=related_queries == i, x=keys, y=zeros))
        related_features.append(related_feature)

    







    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, 500]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, 500]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, 500]

    # Activation
    weight = tf.nn.softmax(scores)  # [B, 1, 500]

    weighted_features = tf.matmul(weight, keys)  # [B, 1, 64]
    weighted_features = tf.squeeze(weighted_features, axis=1) # [B, 64]

    return weight, weighted_features


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


def queries_similarity(queries):
    batch_size, n_attention, emb_size = queries.shape.as_list()
    masks = tf.sequence_mask(list(range(n_attention)), n_attention)
    padding = tf.zeros_like(masks, dtype=tf.float32)

    inner_product = tf.matmul(queries, queries, transpose_b=True)
    inner_product = tf.map_fn(
        lambda x: tf.where(masks, x, padding),
        inner_product,
    )

    similarity = tf.reduce_sum(inner_product)

    return similarity
    

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
    n_attention = params["n_attention"]
    dropout = params["dropout"]
    l2 = params["l2"]
    lr = params["lr"]

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
        jds_emb = tf.nn.embedding_lookup(word_emb, jds)
        jds_conv = cnn(jds_emb, conv_size)
        cvs_emb = tf.nn.embedding_lookup(word_emb, cvs)
        cvs_conv = cnn(cvs_emb, conv_size)

    with tf.variable_scope("user_idx"):
        job_emb = tf.Variable(
            initial_value=tf.random_normal(
                shape=(n_job, n_attention, emb_size),
                stddev=1 / n_word ** (1 / 2),
            ),
            name="job_emb",
        )
        j_queries = tf.nn.embedding_lookup(job_emb, jids)

        person_emb = tf.Variable(
            initial_value=tf.random_normal(
                shape=(n_person, n_attention, emb_size),
                stddev=1 / n_word ** (1 / 2),
            ),
            name="person_emb",
        )        
        p_queries = tf.nn.embedding_lookup(person_emb, pids)

    with tf.variable_scope("attention"):
        jd_weights, jd_weighted_vecs = dynamic_attention(j_queries, jds_conv, jd_lens)
        cv_weights, cv_weighted_vecs = dynamic_attention(p_queries, cvs_conv, cv_lens)

    j_emb = tf.reduce_mean(j_queries, axis=-2)
    p_emb = tf.reduce_mean(p_queries, axis=-2)
    # j_emb, j_variance = tf.nn.moments(j_queries, axes=-2)
    # p_emb, p_variance = tf.nn.moments(p_queries, axes=-2)

    with tf.variable_scope("pooling"):
        jd_global_vecs = tf.reduce_max(jds_conv, axis=1)
        cv_global_vecs = tf.reduce_max(cvs_conv, axis=1)

    features = tf.concat(
        values=[j_emb, jd_global_vecs, jd_weighted_vecs, p_emb, cv_global_vecs, cv_weighted_vecs],
        axis=-1,
    )

    with tf.variable_scope("mlp"):
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

    with tf.variable_scope("loss"):
        fit_label = labels[:, 2]
        loss = tf.losses.log_loss(
            labels=fit_label,
            predictions=tf.squeeze(probs),
        )

        with tf.variable_scope("aux_loss"):
            semantic_features = tf.concat(
                values=[jd_global_vecs, cv_global_vecs],
                axis=-1,
            )
            semantic_prob = tf.sigmoid(mlp(
                semantic_features,
                emb_dim=emb_size,
                dropout=dropout,
                training=(mode == tf.estimator.ModeKeys.TRAIN)
            ))
            semantic_loss = tf.losses.log_loss(
                labels=fit_label,
                predictions=tf.squeeze(semantic_prob),
            )

            jc_label = labels[:, 0]
            jc_features = tf.concat(
                values=[j_emb, cv_weighted_vecs],
                axis=-1,
            )
            jc_prob = tf.sigmoid(mlp(
                jc_features,
                emb_dim=emb_size,
                dropout=dropout,
                training=(mode == tf.estimator.ModeKeys.TRAIN)
            ))
            jc_loss = tf.losses.log_loss(
                labels=jc_label,
                predictions=tf.squeeze(jc_prob)
            )

            cj_label = labels[:, 1]
            cj_features = tf.concat(
                values=[p_emb, jd_weighted_vecs],
                axis=-1,
            )
            cj_prob = tf.sigmoid(mlp(
                cj_features,
                emb_dim=emb_size,
                dropout=dropout,
                training=(mode == tf.estimator.ModeKeys.TRAIN)
            ))
            cj_loss = tf.losses.log_loss(
                labels=cj_label,
                predictions=tf.squeeze(cj_prob)
            )

            j_queries_simi = queries_similarity(j_queries)
            p_queries_simi = queries_similarity(p_queries)

            loss = loss + semantic_loss + jc_loss + cj_loss + j_queries_simi + p_queries_simi

        if l2:
            l2_params = [
                jds_emb,
                cvs_emb,
                j_queries,
                p_queries
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


def input_fn(filenames, batch_size=32, shuffle=0):
    print("parsing", filenames)
    def _parse_fn(record):
        features = {
            "labels": tf.FixedLenFeature([3], tf.int64),
            "jids": tf.FixedLenFeature([], tf.int64),
            "jds": tf.FixedLenFeature([500], tf.int64),
            "jd_lens": tf.FixedLenFeature([], tf.int64),
            "pids": tf.FixedLenFeature([], tf.int64),
            "cvs": tf.FixedLenFeature([500], tf.int64),
            "cv_lens": tf.FixedLenFeature([], tf.int64),
        }
        parsed = tf.parse_single_example(record, features)
        labels = parsed.pop("labels")
        return parsed, labels
    dataset = tf.data.TFRecordDataset(filenames).map(
        _parse_fn, num_parallel_calls=multiprocessing.cpu_count()
    )
    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle)
    itetator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = itetator.get_next()
    return batch_features, batch_labels


if __name__ == "__main__":
    tf.enable_eager_execution()
    batch_features, batch_labels = input_fn(
        "./data/multi_data7_tech/multi_data7_tech.train2.tfrecord",
        batch_size=32,
    )
    print(batch_features)
    print(batch_labels)

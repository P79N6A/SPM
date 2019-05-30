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


def attention(queries: tf.Tensor, keys: tf.Tensor, keys_masks):
    """
    :param queries: [B, 1, H]
    :param keys: [B, T, H]
    :param keys_masks: [B, 1, T]
    :return: 2d array
    """
    d = queries.shape.as_list()[-1]
    # queries = tf.expand_dims(queries, axis=1)  # [B, 1, H]
    scores = tf.divide(
        tf.matmul(queries, keys, transpose_b=True),  # [B, 1, T]
        tf.sqrt(float(d)))

    # Mask
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(keys_masks, scores, paddings)  # [B, 1, T]

    # Activation
    weight = tf.nn.softmax(scores)  # [B, 1, T]

    weighted_features = tf.matmul(weight, keys)  # [B, 1, H]
    # weighted_features = tf.squeeze(weighted_features, axis=1)
    return weighted_features


def multi_attention(queries: tf.Tensor, n_attention:int, keys: tf.Tensor, keys_length: tf.Tensor, cross=0):
    """
    :param queries: [B, 64]
    :param keys: [B, 500, 64]
    :param keys_length: [B]
    :return: 2d array
    """
    batch_size, emb_dim = queries.shape.as_list()

    views = []
    for head in range(n_attention):
        view = tf.layers.dense(
            queries,
            units=emb_dim,
            activation=tf.nn.relu,
        )  # [B, 64]
        view = tf.expand_dims(view, axis=1)
        views.append(view)
    multi_queries = tf.concat(views, axis=1)

    scores = tf.divide(
        tf.matmul(multi_queries, keys, transpose_b=True),
        emb_dim ** (1 / 2),
    )  # [B, 3, 500]
    words_cls = tf.math.argmax(scores, axis=-2)  # [B, 500]

    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, 500]
    paddings = tf.zeros_like(words_cls)
    paddings = tf.cast(paddings, tf.bool)

    multi_views = []
    for head in range(n_attention):
        related_feature_index = tf.equal(words_cls, head)  # [B, 500]
        related_feature_index = tf.where(key_masks, related_feature_index, paddings)  # [B, 500]
        related_feature_index = tf.expand_dims(related_feature_index, axis=-2)  # [B, 1, 500]
        view = views[head]  # [B, 64]
        vec_of_view = attention(view, keys, related_feature_index)
        if cross:
            vec_of_view = pnn([view, vec_of_view])
        multi_views.append(vec_of_view)
    multi_views = tf.concat(multi_views, axis=-1)
    return multi_views


def pnn(features):
    cross_features = tf.einsum(
        'api,apj->apij',
        tf.concat(features[1:], axis=1),
        tf.concat(features[:-1], axis=1)
    )
    cross_features = tf.keras.layers.Flatten()(cross_features)
    return cross_features


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


def cosin(doc1_vector, doc2_vector):
    norms1 = tf.sqrt(tf.reduce_sum(tf.square(doc1_vector), 1, keep_dims=False))
    norms2 = tf.sqrt(tf.reduce_sum(tf.square(doc2_vector), 1, keep_dims=False))
    dot = tf.reduce_sum(doc1_vector * doc2_vector, 1)
    cos_dis = tf.divide(dot, (norms1 * norms2))
    similarity = tf.divide((cos_dis + 1.0), 2.0, name="similarity")
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
    w2v_pre = params["w2v_pre"]
    cross = params["cross"]

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
        jds_conv = cnn(jds_emb, conv_size)
        cvs_emb = tf.nn.embedding_lookup(word_emb, cvs)
        cvs_conv = cnn(cvs_emb, conv_size)

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

    with tf.variable_scope("attention"):
        jd_weighted_vecs = multi_attention(j_emb, n_attention, jds_conv, jd_lens, cross)
        cv_weighted_vecs = multi_attention(p_emb, n_attention, cvs_conv, cv_lens, cross)

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

        with tf.variable_scope("mf_loss"):
            mf_predict = cosin(j_emb, p_emb)
            mf_loss = tf.losses.log_loss(
                labels=fit_label,
                predictions=mf_predict,
            )
        loss += mf_loss

        with tf.variable_scope("semantic_loss"):
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
        loss += semantic_loss

        with tf.variable_scope("jc_loss"):
            jc_label = labels[:, 0]
            jc_features = tf.concat(
                values=[j_emb, cv_weighted_vecs, jd_global_vecs, cv_global_vecs],
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
        loss += jc_loss

        with tf.variable_scope("cj_loss"):
            cj_label = labels[:, 1]
            cj_features = tf.concat(
                values=[p_emb, jd_weighted_vecs, jd_global_vecs, cv_global_vecs],
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

        loss += cj_loss

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

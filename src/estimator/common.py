# -*- coding: utf-8 -*-
import tensorflow as tf
import multiprocessing


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


if __name__ == "__main__":
    tf.enable_eager_execution()
    batch_features, batch_labels = input_fn(
        "./data/multi_data7_tech/multi_data7_tech.train2.tfrecord",
        batch_size=32,
    )
    print(batch_features)
    print(batch_labels)

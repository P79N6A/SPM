# -*- coding: utf-8 -*-
import tensorflow as tf
import sys


def data_format(data):
    data = data.strip()
    data = data.split('\001')
    data = data[:-1]
    data = '\001'.join(data)
    return data


if __name__ == "__main__" :
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        tf.saved_model.loader.load(sess, ['serve'], sys.argv[1])
        writer = tf.summary.FileWriter('board')
        writer.add_graph(sess.graph)
        with open("./data/sample-10k.test") as f:
            data = f.readlines()
            data = [data_format(line) for line in data]
        dataiter = sess.graph.get_operation_by_name('MakeIterator')
        sess.run(dataiter, feed_dict={"serving_input:0": data})
        # logits_tensor = sess.graph.get_tensor_by_name('dnn/logits/BiasAdd:0')
        logits_tensor = sess.graph.get_tensor_by_name('output')
        pre = sess.run(logits_tensor)
        print(pre)
 


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


# 特征名和初始值
COLUMNS_DEFAULT = [
    # job active features num
    ("detail_rate", 0.0),
    ("addf_rate", 0.0),
    ("resp_rate", 0.0),
    ("listcount", 0),
    ("det_count", 0),
    ("addfcount", 0),
    ("chatcount", 0),
    ("activeaddfcount", 0),
    ("freshlist_cnt", 0),
    ("detgeek_cnt", 0),
    ("sent3t5_cnt", 0),
    ("sentlg2_cnt", 0),
    ("retappcount", 0),
    ("freshcount", 0),

    # job basic features num
    ("j_high_salary", 0),
    ("j_low_salary", 0),

    # geek active features num
    ("geek_active_score", 0.0),
    ("addf_num_1d7", 0),
    ("suc_num_1d7", 0),
    ("suc_num_1d30", 0),
    ("tot_sec_1d7", 0),
    ("addf_interval", 0),
    ("suc_interval", 0),
    ("geek_chat_s2_num_1d", 0),
    ("geek_chat_s2_num_1d7", 0),
    ("geek_chat_s5_num_1d7", 0),
    ("geek_notify_num_1d3", 0),
    ("geek_ret_num_1d3", 0),
    ("geek_added_num_day", 0),

    # geek basic features num
    ("work_years", 0),
    ("apply_status", 0),
    ("g_high_salary", 0),
    ("g_low_salary", 0),
    ("mean_salary", 0.0),

    # content features num
    ("page", 0),
    ("rank", 0),

    # job features cate
    ("boss_l1code", ''),
    ("boss_l2code", ''),
    ("b_positioncode", ''),
    ("b_citycode", ''),

    # job basic features cate
    ("com_scale", ''),
    ("com_stage", ''),
    ("area_id", ''),
    ("j_company_level", ''),

    # geek features cate
    ("g_positioncode", ''),
    ("g_citycode", ''),

    # geek basic features cate
    ("degree", ''),
    ("g_company_level", ''),
    ("school_level", ''),
    ("gender", ''),

    # label
    ("label", ''),
]


# 数值特征 用于分桶处理
NUM_FEATURES = [
    # job active features num
    "detail_rate",
    "addf_rate",
    "resp_rate",
    "listcount",
    "det_count",
    "addfcount",
    "chatcount",
    "activeaddfcount",
    "freshlist_cnt",
    "detgeek_cnt",
    "sent3t5_cnt",
    "sentlg2_cnt",
    "retappcount",
    "freshcount",

    # job basic features num
    "j_high_salary",
    "j_low_salary",

    # geek active features num
    "geek_active_score",
    "addf_num_1d7",
    "suc_num_1d7",
    "suc_num_1d30",
    "tot_sec_1d7",
    "addf_interval",
    "suc_interval",
    "geek_chat_s2_num_1d",
    "geek_chat_s2_num_1d7",
    "geek_chat_s5_num_1d7",
    "geek_notify_num_1d3",
    "geek_ret_num_1d3",
    "geek_added_num_day",

    # geek basic features num
    "work_years",
    "apply_status",
    "g_high_salary",
    "g_low_salary",
    "mean_salary",

    # content features num
    "page",
    "rank",
]

# 类别特征 用于哈希处理
CATE_FEATURES = [
    # job features cate
    "boss_l1code",
    "boss_l2code",
    "b_positioncode",
    "b_citycode",

    # job basic features cate
    "com_scale",
    "com_stage",
    "area_id",
    "j_company_level",

    # geek features cate
    "g_positioncode",
    "g_citycode",

    # geek basic features cate
    "degree",
    "g_company_level",
    "school_level",
    "gender",

]


def parse_line(line):
    '''
    处理数据文件或string tensor，划分字符串并对应到特征名
    :param line: \001分割的特征字符串，例如hive中的一条查询结果
    :return: 一个 {特征名：Tensor}字典，一个标签tensor
    '''
    columns, defaults = zip(*COLUMNS_DEFAULT)
    defaults = [[x] for x in defaults]
    line = tf.regex_replace(line, '.N', '')
    fields = tf.decode_csv(
        records=line,
        record_defaults=defaults,
        field_delim='\001',
    )
    features = dict(zip(columns, fields))
    # features = {k: features[k] for k in NUM_FEATURES + ["label"]}
    label = features.pop("label")
    label = tf.not_equal(label, "list")
    label = tf.cast(label, tf.int32)
    return features, label


def input_fn(data_path, batch_size, n_epoch=1):
    '''
    用于训练和评测的输入函数
    :param data_path: 数据目录
    :param batch_size:
    :param n_epoch:
    :return: tf.dataset对象
    '''
    dataset = tf.data.TextLineDataset(data_path)
    dataset = dataset.map(parse_line)
    dataset = dataset.shuffle()
    dataset = dataset.repeat(n_epoch)
    dataset = dataset.batch(batch_size)
    return dataset


def serving_parse_line(line):
    '''
    用于服务的特征处理方法
    :param line: \001分割的特征字符串，例如hive中的一条查询结果
    :return: 一个 {特征名：Tensor}字典
    '''
    columns, defaults = zip(*COLUMNS_DEFAULT[:-1])
    defaults = [[x] for x in defaults]
    line = tf.regex_replace(line, '.N', '')
    fields = tf.decode_csv(
        records=line,
        record_defaults=defaults,
        field_delim='\001',
    )
    features = dict(zip(columns, fields))
    return features


def serving_input_receiver_fn_file(data_path, batch_size):
    '''
    从文本文件产生输入
    :param data_path:
    :param batch_size:
    :return:
    '''
    dataset = tf.data.TextLineDataset(data_path)
    dataset = dataset.map(serving_parse_line)
    dataset = dataset.batch(batch_size)
    dataset = dataset.make_one_shot_iterator()
    data = dataset.get_next()
    return tf.estimator.export.ServingInputReceiver(data, data)


def serving_input_receiver_fn_from_tensor(batch_size):
    '''
    从一个string tensor输入数据，加载模型后，通过"serving_input"占位符输入数据
    :param batch_size:
    :return:
    '''
    sample_tensors = tf.placeholder(dtype=tf.string, name='serving_input')
    dataset = tf.data.Dataset.from_tensor_slices(sample_tensors)
    dataset = dataset.map(serving_parse_line)
    dataset = dataset.batch(batch_size)
    dataset = dataset.make_initializable_iterator(shared_name="dataset")
    data = dataset.get_next()
    return tf.estimator.export.ServingInputReceiver(data, sample_tensors)


def serving_input_receiver_fn_line():
    '''
    接收一条样本输入，加载模型后，通过"serving_input"占位符输入数据
    '''
    serialized_tf_example = tf.placeholder(dtype=tf.string, name='serving_input')
    features = serving_parse_line(serialized_tf_example)
    return tf.estimator.export.ServingInputReceiver(features, serialized_tf_example)


def generate_feature_columns(emb_dim, hash_size_array, boundaries_array):
    '''
    构建feature columns, 数值特征分桶+嵌入，类别特征哈希+嵌入
    :param emb_dim: 嵌入维度
    :param hash_size_array: 哈希空间
    :param boundaries_array: 分桶阈值
    :return: [tf.feature_column,]
    '''
    feature_columns = list()
    boundaries_dic = dict(zip(NUM_FEATURES, boundaries_array))
    for col in NUM_FEATURES:
        boundaries = list(boundaries_dic[col])
        for i in range(len(boundaries)-1):
            if boundaries[i] >= boundaries[i+1]:
                boundaries[i+1] = boundaries[i] + 1e-8
        col_features = tf.feature_column.numeric_column(col)
        col_features = tf.feature_column.bucketized_column(col_features, boundaries)
        col_features = tf.feature_column.embedding_column(col_features, emb_dim)
        feature_columns.append(col_features)
    hash_size_dic = dict(zip(CATE_FEATURES, hash_size_array))
    for col in CATE_FEATURES:
        hash_size = hash_size_dic[col]
        col_features = tf.feature_column.categorical_column_with_hash_bucket(
            col, hash_bucket_size=hash_size)
        col_features = tf.feature_column.embedding_column(col_features, emb_dim)
        feature_columns.append(col_features)
    return feature_columns


def bucket(fp, n_bucket, i_field):
    '''
    计算分桶阈值
    :param fp: 数据路径
    :param n_bucket: 分桶数量
    :param i_field: 特征序号
    :return:
    '''
    def _parse(line):
        data = line.split('\001')
        data = data[i_field]
        data = data.replace("\\N", '0')
        data = float(data)
        return data
    with open(fp) as f:
        data = [_parse(x) for x in f]
    boundaries = [p / n_bucket * 100 for p in range(1, n_bucket)]
    boundaries = [np.percentile(data, x) for x in boundaries]
    return boundaries


if __name__ == "__main__":
    fp = "./data/sample-1m.train"
    n_num_features = 36
    n_bucket = 10

    boundaries_list = list()
    for i in tqdm(range(n_num_features)):
        boundaries = bucket(fp, n_bucket, i)
        boundaries_list.append(boundaries)

    boundaries_frame = pd.DataFrame(boundaries_list)
    boundaries_frame.to_csv("./data/sample-10k.boundaries.csv", index=False, header=False)

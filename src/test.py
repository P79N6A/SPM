import tensorflow as tf
import numpy as np

fpout = "./data/multi_data7_tech/multi_data7_tech.tfrecord"

tfrecord_out = tf.python_io.TFRecordWriter(fpout)

labels = np.random.randint(1, size=3)
job_id = 1
jd_lens = 100
jd = np.random.randint(100, size=500)
exp_id = 1
cv = np.random.randint(100, size=500)
cv_lens = 100


feature = {
    "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
    "jids": tf.train.Feature(int64_list=tf.train.Int64List(value=[job_id])),
    "jds": tf.train.Feature(int64_list=tf.train.Int64List(value=jd)),
    "jd_lens": tf.train.Feature(int64_list=tf.train.Int64List(value=[jd_lens])),
    "pids": tf.train.Feature(int64_list=tf.train.Int64List(value=[exp_id])),
    "cvs": tf.train.Feature(int64_list=tf.train.Int64List(value=cv)),
    "cv_lens": tf.train.Feature(int64_list=tf.train.Int64List(value=[cv_lens])),
}
example = tf.train.Example(
    features=tf.train.Features(feature=feature)
)
serialized = example.SerializeToString()
tfrecord_out.write(serialized)
tfrecord_out.close()

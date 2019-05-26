# -*- coding: utf-8 -*-
import re
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json


class MixData:
    def __init__(self, fpin, fpout, wfreq, doc_len, pre_w2v=""):
        self.fpin = fpin
        self.doc_len = doc_len

        exp_features_names = [
            'expect_id',
            'doc_token',
            'skills',
        ]
        job_features_names = [
            'job_id',
            'doc_token',
            'skills',
        ]

        fps = [
            '{}.profile.job'.format(fpin),
            '{}.profile.expect'.format(fpin)]

        if pre_w2v:
            pre_words = self.load_pretrain(pre_w2v)
            self.word_dict = self.build_dict(fps, wfreq, pre_words)
            self.dump_w2v(
                pre_w2v,
                self.word_dict,
                dump_fp="{}.w2v".format(fpout),
            )
        else:
            self.word_dict = self.build_dict(fps, wfreq)

        with open("{}.words.json".format(fpout), "w", encoding="utf8") as f:
            json.dump(self.word_dict, f, ensure_ascii=False, indent=2)

        with open("{}.words.tsv".format(fpout), "w", encoding="utf8") as f:
            f.write("Index\tLabel\n")
            for k, v in self.word_dict.items():
                line = "{}\t{}\n".format(v, k)
                f.write(line)
        self.exp_to_row, self.exp_docs, self.exp_doc_lens, self.exp_doc_raw = self.build_features(
            fp='{}.profile.expect'.format(fpin),
            feature_name=exp_features_names,
        )
        self.job_to_row, self.job_docs, self.job_doc_lens, self.job_doc_raw = self.build_features(
            fp='{}.profile.job'.format(fpin),
            feature_name=job_features_names,
        )
        print("num of words: {}".format(len(self.word_dict)))
        print("num of person: {}".format(len(self.exp_to_row)))
        print("num of job: {}".format(len(self.job_to_row)))
        with open("{}.param.json".format(fpout), "w") as f:
            json.dump(
                {
                    "n_word": len(self.word_dict),
                    "n_person": len(self.exp_to_row),
                    "n_job": len(self.job_to_row),
                },
                f,
            )

    @staticmethod
    def load_pretrain(fp):
        if not fp:
            return False
        words = set(["__pad__", "__unk__"])
        with open(fp, encoding="utf8") as f:
            for line in f:
                word = line.split()[0]
                words.add(word)
        return words

    @staticmethod
    def build_dict(fps, w_freq, word_set=False):
        words = []
        for fp in fps:
            with open(fp, encoding="utf8") as f:
                for line in tqdm(f):
                    line = line.strip().split('\001')[-3:]
                    line = '\t'.join(line)
                    line = re.split("[ \t]", line)
                    words.extend(line)
        words_freq = collections.Counter(words)
        word_list = [k for k, v in words_freq.items() if v >= w_freq]
        word_list = ['__pad__', '__unk__'] + word_list
        if word_set:
            word_list = [word for word in word_list if word in word_set]
        word_dict = {k: v for v, k in enumerate(word_list)}
        print('n_words: {}'.format(len(word_dict)), len(word_list))
        return word_dict

    @staticmethod
    def dump_w2v(pre_w2v, word_dict, dump_fp):
        print("dump w2v ing ...")
        # with open(dump_fp, 'w', encoding="utf8") as fout:
        #     with open(pre_w2v) as fin:
        #         for line in tqdm(fin):
        #             word = line.split()[0]
        #             if word not in word_dict:
        #                 continue
        #             fout.write(line)
        with open(pre_w2v, encoding="utf8") as fin:
            line = fin.readline().strip()
            emb_dim = len(line.split()) - 1
        word_vecs = [("0 " * emb_dim)[:-1]] * len(word_dict)
        print("loading w2v")
        with open(pre_w2v, encoding="utf8") as fin:
            for line in tqdm(fin):
                data = line.split()
                word = data[0]
                if word not in word_dict:
                    continue
                idx = word_dict[word]
                word_vecs[idx] = " ".join(data[1:])
        print("writing vec")
        with open(dump_fp, 'w') as f:
            f.write("\n".join(word_vecs))
        return

    def build_features(self, fp, feature_name):
        n_feature = len(feature_name)
        print('split raw data ...')
        with open(fp, encoding="utf8") as f:
            data_list = []
            for line in tqdm(f):
                features = line.split('\001')
                if len(features) != n_feature:
                    continue
                # id
                cid = features[0]
                features_dict = dict(zip(feature_name, features))
                # text feature
                words = features_dict["doc_token"].strip()
                word_ids, true_len = self.doc1d(words, self.doc_len)
                # raw text
                doc = features_dict['doc_token']
                # reduce
                data_list.append([cid, word_ids, true_len, doc])
        ids_list, word_ids, true_lens, doc = list(zip(*data_list))
        # id
        id_to_row = {k: v for v, k in enumerate(ids_list)}
        # doc
        word_ids = np.array(word_ids)
        true_lens = np.array(true_lens)
        return id_to_row, word_ids, true_lens, doc

    def doc1d(self, sent, sent_len):
        if type(sent) == str:
            sent = sent.strip().split(' ')
        sent = [self.word_dict.get(word, 0) for word in sent][:sent_len]
        sent_len_true = len(sent)
        if len(sent) < sent_len:
            sent += [0] * (sent_len - len(sent))
        return sent, sent_len_true

    def feature_lookup(self, idstr, idtype="pad"):
        if idtype == 'expect':
            row = self.exp_to_row[idstr]
            docs = self.exp_docs[row]
            doc_lens = self.exp_doc_lens[row]
            raw_docs = self.exp_doc_raw[row]
        elif idtype == "job":
            row = self.job_to_row[idstr]
            docs = self.job_docs[row]
            doc_lens = self.job_doc_lens[row]
            raw_docs = self.job_doc_raw[row]
        else:
            row = 0
            docs = np.zeros_like(self.job_docs[0])
            doc_lens = 0
            raw_docs = np.zeros_like(self.job_doc_raw[0])
        return [row, docs, doc_lens, raw_docs]

    def tfrecord_generate(self, fpin, fpout):
        tfrecord_out = tf.python_io.TFRecordWriter(fpout)
        with open(fpin) as f:
            for line in tqdm(f):
                eid, jid, *labels = line.strip().split('\001')
                labels = [int(x) for x in labels]
                if jid not in self.job_to_row or eid not in self.exp_to_row:
                    print('loss id')
                    continue
                job_id, jd, jd_lens, jd_raw = self.feature_lookup(jid, 'job')
                exp_id, cv, cv_lens, cv_raw = self.feature_lookup(eid, 'expect')
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
        return


if __name__ == '__main__':
    import os
    print("work directory: ", os.getcwd())

    dataset = "multi_data7_300k"
    dataout = "multi_data7_300k_pre"
    mix_data = MixData(
        fpin='./data/{0}/{0}'.format(dataset),
        fpout="./data/{0}/{0}".format(dataout),
        wfreq=10,
        doc_len=500,
        pre_w2v="./data/Tencent_AILab_ChineseEmbedding.txt"
    )

    fpin = "./data/{0}/{0}.train2".format(dataset)
    fpout = "./data/{0}/tfrecord/{0}.train2.tfrecord".format(dataout)
    mix_data.tfrecord_generate(fpin, fpout)

    fpin = "./data/{0}/{0}.test2".format(dataset)
    fpout = "./data/{0}/tfrecord/{0}.test2.tfrecord".format(dataout)
    mix_data.tfrecord_generate(fpin, fpout)



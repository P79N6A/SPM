import jieba
import re
from tqdm import tqdm
import argparse
import pandas as pd
import os


def get_structs(doc):
    paras = re.split('[;；。!?？！]', doc)
    paras = [jieba.cut(para) for para in paras]
    paras = [' '.join(para) for para in paras]
    paras = '\t'.join(paras)
    return paras


def nlp_feature(data):
    idx = data[0]
    doc = data[-1]
    requirements = get_structs(doc)
    doc = doc.replace(' ', '')
    doc = re.sub("[。]+", "。", doc)
    words = jieba.cut(doc)
    words = ' '.join(words)
    new_line = "{}\001{}\001{}".format(idx, words, requirements)
    return new_line


def nlp_features(fpin, fpout, id_pool):
    with open(fpin, encoding="utf8") as fin:
        datain = fin.readlines()
        dataout = []
        for line in tqdm(datain):
            line = line.strip()
            data = line.split('\001')
            idx = data[0]
            if idx not in id_pool:
                continue
            new_line = nlp_feature(data)
            dataout.append(new_line)
    with open(fpout, 'w', encoding="utf8") as fout:
        fout.write('\n'.join(dataout))
    return


def get_id_pool(path):
    train_frame = pd.read_csv(
        "./data/{}/{}.train2".format(path, path),
        sep='\001',
        header=None,
        dtype=str,
    )
    test_frame = pd.read_csv(
        "./data/{}/{}.test2".format(path, path),
        sep='\001',
        header=None,
        dtype=str,
    )
    frame = pd.concat([train_frame, test_frame])
    exp_pool = set(frame.iloc[:, 0])
    job_pool = set(frame.iloc[:, 1])
    print("n_job: {}, n_exp: {}".format(len(job_pool), len(exp_pool)))
    return exp_pool, job_pool


if __name__ == "__main__":
    print("work directory: ", os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="multi_data7")
    args = parser.parse_args()

    exp_pool, job_pool = get_id_pool(args.data)

    nlp_features(
        fpin="../Data/{}/{}.profile.job".format(args.data, args.data),
        fpout="./data/{}/{}.profile.job".format(args.data, args.data),
        id_pool=job_pool,
    )
    nlp_features(
        fpin="../Data/{}/{}.profile.expect".format(args.data, args.data),
        fpout="./data/{}/{}.profile.expect".format(args.data, args.data),
        id_pool=exp_pool,
    )


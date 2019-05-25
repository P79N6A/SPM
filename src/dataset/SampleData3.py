from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import os


def calibration(frame: pd.DataFrame, rate, random_neg=False):
    posi_frame = frame[frame['fit_label'] == '1']
    nega_frame = frame[frame['fit_label'] == '0']
    job_posi_frame = frame[frame["jc_label"] == '1']
    exp_posi_frame = frame[frame["cj_label"] == '1']
    print("num positive: {}".format(len(posi_frame)))
    print("num negative: {}".format(len(nega_frame)))
    print("num job posi: {}".format(len(job_posi_frame)))
    print("num person posi: {}".format(len(exp_posi_frame)))
    if random_neg:
        cids = list(set(frame["cid"]))
        jids = list(set(frame["jid"]))
        #todo

    if len(posi_frame) < rate * len(nega_frame):
        frame = pd.concat([
            posi_frame,
            nega_frame.sample(n=int(len(posi_frame) / rate))
        ])
        col = frame.columns
        frame = pd.DataFrame(np.random.permutation(frame.values))
        frame.columns = col
    print("num negative: {}".format(len(frame) - len(posi_frame)))
    job_posi_frame = frame[frame["jc_label"] == '1']
    print("num job posi: {}".format(len(job_posi_frame)))
    exp_posi_frame = frame[frame["cj_label"] == '1']
    print("num person posi: {}".format(len(exp_posi_frame)))
    return frame


def sample_new_pair(test_frame: pd.DataFrame, train_frame: pd.DataFrame):
    train_frame = train_frame[train_frame["fit_label"] == '1']
    cv_set = set(train_frame["cid"])
    jd_set = set(train_frame["jid"])
    test_data = []
    for row in tqdm(test_frame.values):
        cid, jid, *labels = row
        # if cid in cv_set and jid in jd_set:
        if cid in cv_set or jid in jd_set:
            continue
        test_data.append(row)
    test_frame = pd.DataFrame(test_data, columns=train_frame.columns)
    return test_frame


if __name__ == '__main__':
    print("work directory: ", os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--datain', default="multi_data7")
    parser.add_argument('--dataout', default="multi_data7_300k")
    parser.add_argument('--rand_sample', type=int, default=0)
    parser.add_argument("--new", type=int, default=0)
    args = parser.parse_args()

    fpin="../Data/{}/{}".format(args.datain, args.datain)
    fpout="./data/{}/{}".format(args.dataout, args.dataout)

    train_frame = pd.read_csv(
        '{}.train'.format(fpin),
        sep='\001',
        header=None,
        dtype=str,
    )
    train_frame.columns = ['cid', 'jid', 'jc_label', "cj_label", 'fit_label']
    train_frame = calibration(train_frame, 0.1)

    if args.rand_sample:
        train_frame, test_frame = train_frame.iloc[:-20000], train_frame.iloc[-20000:]
    else:
        test_frame = pd.read_csv(
            '{}.test'.format(fpin),
            sep='\001',
            header=None,
            dtype=str,
        )
        test_frame.columns = ['cid', 'jid', 'jc_label', "cj_label", 'fit_label']
        if args.new:
            train_frame = train_frame.iloc[:300000]
            test_frame = sample_new_pair(test_frame, train_frame)
        test_frame = calibration(test_frame, 0.1)

    train_frame = train_frame.iloc[:300000]

    test_frame = test_frame.iloc[:50000]

    train_frame.to_csv('{}.train2'.format(fpout), sep='\001', header=None, index=False)
    test_frame.to_csv('{}.test2'.format(fpout), sep='\001', header=None, index=False)
    # train_frame[["cid", "jid", "fit_label"]].to_csv('{}.train'.format(args.dataout), sep='\001', header=None, index=False)
    # test_frame[["cid", "jid", "fit_label"]].to_csv('{}.test'.format(args.dataout), sep='\001', header=None, index=False)




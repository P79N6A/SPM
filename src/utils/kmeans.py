from sklearn import cluster
from tqdm import tqdm
import pandas as pd


def load_w2v(fp_pre):
    data = []
    with open(fp_pre) as f:
        for line in tqdm(f):
            line = line.strip().split()
            if len(line) != 200:
                line = line[1:]
            line = [float(x) for x in line]
            data.append(line)
    return data


def multiple_cluster(data):
    result = []
    for n in range(1, 10, 2):
        model = cluster.KMeans(n_clusters=n)
        model.fit(data)
        y = model.predict(data)
        result.append(y)
    return result


if __name__ == "__main__":
    data = "multi_data7_300k_pre"
    fp_pre = "./data/{0}/{0}.w2v".format(data)

    w2v = load_w2v(fp_pre)
    result = multiple_cluster(w2v)

    fp_cls = "./data/{0}/{0}.word_cls".format(data)
    df_cls = pd.DataFrame(result).T
    df_cls.to_csv(fp_cls, index=False, header=None, sep=' ')

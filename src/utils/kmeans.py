from sklearn import cluster
from tqdm import tqdm

if __name__ == "__main__":
    data = "multi_data7_300k_pre"
    fp_pre = "./data/{0}/{0}.w2v".format(data)
    data = []
    with open(fp_pre) as f:
        for line in tqdm(f):
            line = line.strip().split()
            if len(line) != 200:
                line = line[1:]
            line = [float(x) for x in line]
            data.append(line)

    result = []
    for n in range(1, 10, 2):
        model = cluster.KMeans(n_clusters=3)
        model.fit(data)
        y = model.predict(data)
        result.append(y)

    with open("./data/multi_data7_300k_pre.")
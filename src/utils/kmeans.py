from sklearn import cluster
import pandas as pd

if __name__ == "__main__":
    data = "multi_data7_300k_pre"
    fp_pre = "./data/{0}/{0}.w2v".format(data)
    data = pd.read_csv(fp_pre, sep=" ", header=None).values
    
    model = cluster.KMeans(n_clusters=3)
    y = model.fit(data)
    print(y)

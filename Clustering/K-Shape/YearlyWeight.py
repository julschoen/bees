from dataset.Dataset import Dataset
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesResampler
import numpy as np
from Clustering.StandardScaler import StandardScaler


def load_data():
    train = Dataset().get_year_weight()
    clean_train = []
    scaler = StandardScaler()
    for row in train:
        r = []
        for arr in row:
            arr = np.array(arr).astype(np.float)
            r.append(np.array(arr).astype(np.float))
        r = np.array(r)
        clean_train.append(r)

    clean_train = np.array(clean_train)
    scaler.fit(clean_train)
    clean_train = scaler.transform(clean_train)

    x_train = to_time_series_dataset(clean_train)
    return TimeSeriesResampler(sz=240).fit_transform(x_train)


def plot(data, labels, cluster_amount):
    colors = {1: (0, 0, 1), 2: (1, 0, 0), 0: (0, 1, 0), 3: (0.5, 0.5, 0), 4: (0, 0.5, 0.5), 5: (0.5, 0, 0.5)}
    cl_cntr = [0] * 5
    for j in range(cluster_amount):
        for i, x in enumerate(data):
            if labels[i] == j:
                cl_cntr[j] += 1
                plt.plot(x, c=colors[j])
    plt.show()
    print(cl_cntr)
    for j in range(cluster_amount):
        for i, x in enumerate(data):
            if labels[i] == j:
                plt.plot(x)
        plt.show()


data = load_data()
cluster_amount = 5
kshape = KShape(n_clusters=cluster_amount, n_init=10, verbose=False)
kshape.fit(data)
labels = kshape.labels_
plot(data, labels, cluster_amount)
# kshape.to_pickle('kshape.p')

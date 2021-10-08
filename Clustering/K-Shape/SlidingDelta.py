from dataset.Dataset import Dataset
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesResampler
import numpy as np
from Clustering.StandardScaler import StandardScaler


def load_data():
    weights = Dataset().get_year_weight(trachttag_delta=True)
    scaler = StandardScaler()
    scaler.fit(weights)
    weights = scaler.transform(weights)
    clean_train = []
    for year in weights:
        data = []
        max_i = 0
        for i in range(0, len(year) - 31, 30):
            max_i = i + 31
            data.append(np.mean(year[i:i + 30]))
        data.append(np.mean(year[max_i:]))
        if len(data) > 1:
            clean_train.append(list(data))
        else:
            clean_train.append(list(data + data))

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
# kshape.to_pickle('kshape_sliding_delta.p')

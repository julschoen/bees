from dataset.Dataset import Dataset
import numpy as np
from Clustering.KMeans import KMeans
from Clustering.StandardScaler import StandardScaler


def load_data():
    data = Dataset().get_year_weight()
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return np.array([weight.pop(len(weight) - 1) for weight in data]).reshape(-1, 1)


km = KMeans(4)
km.fit(load_data())
print(km.score())
km.to_pickle('ending.p')

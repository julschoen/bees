from dataset.Dataset import Dataset
from scipy.stats import skew, kurtosis
import numpy as np
from Clustering.KMeans import KMeans


def load_data(trachttag=False):
    yearly = Dataset().get_year_weight(trachttag_delta=trachttag)
    dataset = []
    for year in yearly:
        mean = np.mean(year)
        stand_dev = np.std(year)
        sk = skew(year)
        kurt = kurtosis(year)
        dataset.append([mean, stand_dev, sk, kurt])
    return dataset


km = KMeans(5)
km.fit(load_data())
print(km.score())
# km.to_pickle('features_year.p')


km = KMeans(4)
km.fit(load_data(trachttag=True))
print(km.score())
# km.to_pickle('features_delta.p')

from dataset.Dataset import Dataset
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
import numpy as np
from Clustering.KMeans import KMeans
from Clustering.StandardScaler import StandardScaler


def load_data():
    years = Dataset().get_year_weight()
    scaler = StandardScaler()
    scaler.fit(years)
    years = scaler.transform(years)
    clean_train = []
    for year in years:
        data = []
        max_i = 0
        for i in range(0, len(year) - 31, 30):
            data.append(np.mean(year[i:i + 30]))
            max_i = i + 31
        data.append(np.mean(year[max_i:]))
        clean_train.append(list(data))
    x_train = to_time_series_dataset(clean_train)
    return TimeSeriesResampler(sz=240).fit_transform(x_train)


km = KMeans(4, True)
km.fit(load_data())
print(km.score())
# km.plot()
# km.to_pickle('sliding_mean.p')

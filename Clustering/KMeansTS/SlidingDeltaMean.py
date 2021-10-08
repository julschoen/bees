from dataset.Dataset import Dataset
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
import numpy as np
from Clustering.KMeans import KMeans
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



km = KMeans(4, time_series=True)
km.fit(load_data())
print(km.score())
#km.plot()
# km.to_pickle('sliding_delta_mean.p')

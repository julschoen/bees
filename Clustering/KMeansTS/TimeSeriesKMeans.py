from dataset.Dataset import Dataset
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesResampler
import numpy as np
from Clustering.StandardScaler import StandardScaler
from Clustering.KMeans import KMeans


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



km = KMeans(5, time_series=True)
km.fit(load_data())
print(km.score())
km.plot()
# km.to_pickle('KmeansYearly.p')

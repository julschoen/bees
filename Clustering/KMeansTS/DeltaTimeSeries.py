from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesResampler
from dataset.Dataset import Dataset
from Clustering.KMeans import KMeans
from Clustering.StandardScaler import StandardScaler


def load_data():
    data = Dataset().get_year_weight(trachttag_delta=True)
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    x_train = to_time_series_dataset(data)
    return TimeSeriesResampler(sz=240).fit_transform(x_train)


km = KMeans(4, time_series=True)
km.fit(load_data())
print(km.score())
# km.plot()
# km.to_pickle('delta.p')

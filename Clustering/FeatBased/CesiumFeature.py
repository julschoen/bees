from cesium import featurize
from dataset.Dataset import Dataset
import numpy as np
from Clustering.KMeans import KMeans


def load_data(selected=False):
    features = ['amplitude', 'stetson_k', 'minimum', 'percent_close_to_median', 'percent_beyond_1_std',
                'maximum', 'median', 'median_absolute_deviation', 'skew', 'max_slope', 'std']
    sel_feat = ['skew', 'stetson_k', 'minimum', 'percent_close_to_median', 'percent_beyond_1_std']

    if selected:
        features = sel_feat

    data = Dataset().get_year_weight()
    dataset = []
    for weight in data:
        ts = featurize.TimeSeries(m=weight)
        feat = featurize.featurize_single_ts(ts, features_to_use=features)
        feat = feat.tolist()
        dataset.append(np.array(feat))
    return dataset


km = KMeans(5)
km.fit(load_data())
print(km.score())
# km.to_pickle('cesium.p')


km = KMeans(4)
km.fit(load_data(selected=True))
print(km.score())
# km.to_pickle('cesium_feature_selection.p')

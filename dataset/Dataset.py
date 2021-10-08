import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
import pickle
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesResampler
import numpy as np
from scipy.stats import skew, kurtosis
from tslearn.clustering import TimeSeriesKMeans, KShape
from cesium import featurize
from Clustering.StandardScaler import StandardScaler


class Dataset(object):

    def __init__(self, path_to_data, path_to_test_data, path_to_delta_data',
                 path_to_delta_data_test, features=None, days=30):
        if features is None:
            features = ["weight", "hive_temperature", "hive_humidity", "wind_speed", "wind_speed_max",
                        "weather_temperature_min", "weather_temperature_max", "weather_temperature",
                        "weather_humidity", "rain", "rain_min", "rain_max"]
        self.days = days
        self.data_path = path_to_data
        self.test_path = path_to_test_data
        self.delta_path = path_to_delta_data
        self.delta_test_path = path_to_delta_data_test
        self.features = features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.std_scaler = StandardScaler()
        self.__fit_std_scaler()
        self.delta_scaler = StandardScaler()
        self.__fit_delta_scaler()

    def __fit_std_scaler(self):
        train = self.get_year_weight()
        clean_train = []
        for row in train:
            r = []
            for arr in row:
                arr = np.array(arr).astype(np.float)
                r.append(np.array(arr).astype(np.float))
            r = np.array(r)
            clean_train.append(r)

        clean_train = np.array(clean_train)
        self.std_scaler.fit(clean_train)

    def __fit_delta_scaler(self):
        train = self.get_year_weight(trachttag_delta=True)
        clean_train = []
        for row in train:
            r = []
            for arr in row:
                arr = np.array(arr).astype(np.float)
                r.append(np.array(arr).astype(np.float))
            r = np.array(r)
            clean_train.append(r)
        clean_train = np.array(clean_train)
        self.delta_scaler.fit(clean_train)

    def get_data(self, test=False, shuffle=False, scaled=True, val_data=0.1):
        directory = self.data_path
        if test:
            directory = self.test_path

        dfs, dfs_delta = self.__load_dataframes(self.features, directory, test=test)
        self.scaler.fit(pd.concat(dfs, ignore_index=True))
        x, y = self.__split_data(dfs, scaled=scaled)

        if shuffle:
            x, y = self.__shuffle(x, y)
        if test:
            return x, y
        cutpoint = int(round(len(x) * val_data))
        x_val, y_val = x[:cutpoint], y[:cutpoint]
        x, y = x[cutpoint:], y[cutpoint:]

        return x, y, x_val, y_val

    def get_cluster_data(self, test=False, shuffle=False, scaled=True, val_data=0.1, cluster_algo='yearly'):
        directory = self.data_path
        if test:
            directory = self.test_path
        dfs, dfs_delta = self.__load_dataframes(self.features, directory, test=test)
        self.scaler.fit(pd.concat(dfs, ignore_index=True))
        cluster_algo = cluster_algo.lower()

        if cluster_algo not in ['yearly', 'ending', 'delta', 'sliding_mean', 'sliding_delta_mean', 'feature_yearly',
                                'feature_delta', 'cesium', 'cesium_selected', 'kshape', 'kshape_delta',
                                'kshape_sliding', 'kshape_sliding_delta']:
            raise Exception('Clustering algorithm must be one of the following: yearly, ending, delta, '
                            'sliding_mean, feature_yearly, feature_delta, sliding_delta_mean or kshape!')
        elif cluster_algo in ['yearly', 'delta', 'sliding_mean', 'sliding_delta_mean', 'kshape', 'kshape_delta',
                              'kshape_sliding', 'kshape_sliding_delta']:

            if cluster_algo == 'yearly':
                kmeans = TimeSeriesKMeans.from_pickle("../Clustering/KMeansTS/KmeansYearly.p")
                labels = []
                for j in range(len(dfs)):
                    df = dfs[j].copy()

                    label = [0]*5

                    weight = list(df["weight"])
                    weight = self.std_scaler.transform(weight)
                    x_train = to_time_series_dataset([weight])
                    x_train = TimeSeriesResampler(sz=240).fit_transform(x_train)

                    label[kmeans.predict(x_train)[0]] = 1

                    labels.append([label])

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'kshape':
                kshape = KShape.from_pickle('../Clustering/K-Shape/kshape.p')
                labels = []
                for j in range(len(dfs)):
                    df = dfs[j].copy()

                    label = [0]*5

                    weight = list(df["weight"])
                    weight = self.std_scaler.transform(weight)
                    x_train = to_time_series_dataset([weight])
                    x_train = TimeSeriesResampler(sz=240).fit_transform(x_train)

                    label[kshape.predict(x_train)[0]] = 1

                    labels.append([label])

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'kshape_delta':
                kshape = KShape.from_pickle('../Clustering/K-Shape/kshape_delta.p')
                labels = []
                for j in range(len(dfs)):
                    df = dfs_delta[j].copy()

                    label = [0] * 5

                    weight = list(df["weight"])
                    weight = self.delta_scaler.transform(weight)
                    x_train = to_time_series_dataset([weight])
                    x_train = TimeSeriesResampler(sz=240).fit_transform(x_train)

                    label[kshape.predict(x_train)[0]] = 1

                    labels.append([label])

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'kshape_sliding':
                kshape = KShape.from_pickle('../Clustering/K-Shape/kshape_sliding.p')
                labels = []
                for j in range(len(dfs)):
                    df = dfs[j].copy()

                    label = [0] * 5

                    weight = list(df["weight"])
                    weight = self.std_scaler.transform(weight)

                    data = []
                    max_i = 0
                    for i in range(0, len(weight) - 31, 30):
                        data.append(np.mean(weight[i:i + 30]))
                        max_i = i + 31
                    data.append(np.mean(weight[max_i:]))

                    x_train = to_time_series_dataset([data])
                    x_train = TimeSeriesResampler(sz=240).fit_transform(x_train)

                    label[kshape.predict(x_train)[0]] = 1

                    labels.append([label])

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'kshape_sliding_delta':
                kshape = KShape.from_pickle('../Clustering/K-Shape/kshape_sliding_delta.p')
                labels = []
                for j in range(len(dfs)):
                    df = dfs_delta[j].copy()

                    label = [0] * 5

                    weight = list(df["weight"])
                    weight = self.delta_scaler.transform(weight)
                    clean_train = []

                    data = []
                    max_i = 0
                    for i in range(0, len(weight) - 31, 30):
                        max_i = i + 31
                        data.append(np.mean(weight[i:i + 30]))
                    data.append(np.mean(weight[max_i:]))
                    if len(data) > 1:
                        clean_train.append(list(data))
                    else:
                        clean_train.append(list(data + data))
                    x_train = to_time_series_dataset(clean_train)
                    x_train = TimeSeriesResampler(sz=240).fit_transform(x_train)

                    label[kshape.predict(x_train)[0]] = 1

                    labels.append([label])

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'sliding_mean':
                kmeans = TimeSeriesKMeans.from_pickle("../Clustering/KMeansTS/sliding_mean.p")
                labels = []
                for j in range(len(dfs)):
                    df = dfs[j].copy()

                    label = [0]*5

                    weight = list(df["weight"])
                    weight = self.std_scaler.transform(weight)

                    data = []
                    max_i = 0
                    for i in range(0, len(weight) - 31, 30):
                        data.append(np.mean(weight[i:i + 30]))
                        max_i = i + 31
                    data.append(np.mean(weight[max_i:]))

                    x_train = to_time_series_dataset([data])
                    x_train = TimeSeriesResampler(sz=240).fit_transform(x_train)

                    label[kmeans.predict(x_train)[0]] = 1

                    labels.append([label])
                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'sliding_delta_mean':
                kmeans = TimeSeriesKMeans.from_pickle("../Clustering/KMeansTS/sliding_delta_mean.p")
                labels = []
                for j in range(len(dfs)):
                    df = dfs_delta[j].copy()

                    label = [0]*5

                    weight = list(df["weight"])
                    weight = self.delta_scaler.transform(weight)
                    clean_train = []

                    data = []
                    max_i = 0
                    for i in range(0, len(weight) - 31, 30):
                        max_i = i + 31
                        data.append(np.mean(weight[i:i + 30]))
                    data.append(np.mean(weight[max_i:]))
                    if len(data) > 1:
                        clean_train.append(list(data))
                    else:
                        clean_train.append(list(data + data))
                    x_train = to_time_series_dataset(clean_train)
                    x_train = TimeSeriesResampler(sz=240).fit_transform(x_train)
                    label[kmeans.predict(x_train)[0]] = 1
                    labels.append([label])
                x, y = self.__split_data(dfs, labels, scaled=scaled)
            else:
                kmeans = TimeSeriesKMeans.from_pickle("../Clustering/KMeansTS/delta.p")
                labels = []
                for j in range(len(dfs)):
                    df = dfs_delta[j].copy()

                    label = [0]*5

                    weight = list(df["weight"])
                    weight = self.delta_scaler.transform(weight)
                    x_train = to_time_series_dataset([weight])
                    x_train = TimeSeriesResampler(sz=240).fit_transform(x_train)

                    label[kmeans.predict(x_train)[0]] = 1

                    labels.append([label])

                x, y = self.__split_data(dfs, labels, scaled=scaled)

        else:
            if cluster_algo == 'cesium':

                features_to_use = ['amplitude', 'stetson_k', 'minimum', 'percent_close_to_median',
                                   'percent_beyond_1_std',
                                   'maximum', 'median', 'median_absolute_deviation', 'skew', 'max_slope', 'std']

                labels = []
                kmeans = pickle.load(
                    open("../Clustering/FeatBased/cesium.p","rb")
                )

                for j in range(len(dfs)):
                    df = dfs[j].copy()

                    label = [0]*5

                    weight = list(df["weight"])
                    ts = featurize.TimeSeries(m=weight)
                    features = dict(featurize.featurize_single_ts(ts, features_to_use=features_to_use))
                    prediction = kmeans.predict([list(features.values())])[0]

                    label[prediction] = 1

                    labels.append(np.array([label]))

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'cesium_selected':
                features_to_use = ['skew', 'stetson_k', 'minimum', 'percent_close_to_median', 'percent_beyond_1_std']

                labels = []
                kmeans = pickle.load(
                    open(
                        "../Clustering/FeatBased/cesium_feature_selection.p",
                        "rb")
                )

                for j in range(len(dfs)):
                    df = dfs[j].copy()

                    label = [0]*5

                    weight = list(df["weight"])
                    ts = featurize.TimeSeries(m=weight)
                    features = dict(featurize.featurize_single_ts(ts, features_to_use=features_to_use))
                    prediction = kmeans.predict([list(features.values())])[0]

                    label[prediction] = 1

                    labels.append(np.array([label]))

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'feature_yearly':

                labels = []
                kmeans = pickle.load(
                    open("../Clustering/FeatBased/features_year.p", "rb")
                )

                for j in range(len(dfs)):
                    df = dfs[j].copy()

                    label = [0]*5

                    weight = np.array(df["weight"])
                    weight = [np.mean(weight), np.std(weight), skew(weight), kurtosis(weight)]
                    prediction = kmeans.predict([weight])[0]
                    label[prediction] = 1

                    labels.append([label])

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            elif cluster_algo == 'feature_delta':

                labels = []
                kmeans = pickle.load(
                    open("../Clustering/FeatBased/features_delta.p", "rb")
                )

                for j in range(len(dfs)):
                    df = dfs_delta[j].copy()

                    label = [0] * 5

                    weight = np.array(df["weight"])
                    weight = [np.mean(weight), np.std(weight), skew(weight), kurtosis(weight)]
                    prediction = kmeans.predict([weight])[0]
                    label[prediction] = 1

                    labels.append([label])

                x, y = self.__split_data(dfs, labels, scaled=scaled)
            else:
                labels = []
                kmeans = pickle.load(
                    open("../Clustering/FeatBased/ending.p", "rb")
                )

                for j in range(len(dfs)):
                    df = dfs[j].copy()

                    label = [0]*5

                    weight = list(df["weight"])
                    weight = self.std_scaler.transform(weight)
                    weight = [weight[len(weight) - 1]]

                    prediction = kmeans.predict([weight])[0]
                    label[prediction] = 1

                    labels.append(np.array([label]))

                x, y = self.__split_data(dfs, labels, scaled=scaled)

        if shuffle:
            x, y = self.__shuffle(x, y)

        if test:
            return np.array(x), np.array(y)

        cutpoint = int(round(len(x) * val_data))
        x, y = x[cutpoint:], y[cutpoint:]
        x_val, y_val = x[:cutpoint], y[:cutpoint]

        return np.array(x), np.array(y), np.array(x_val), np.array(y_val)

    def get_year_weight(self, trachttag_delta=False):

        if trachttag_delta:
            dfs, dfs_detla = self.__load_dataframes(["weight"],self.path_to_delta_data)
            weight = [list(w["weight"]) for w in dfs]
        else:
            dfs, _ = self.__load_dataframes(["weight", "hive_temperature", "hive_humidity", "datetime"], self.data_path)
            weight = [list(w["weight"]) for w in dfs]
        return weight

    def __load_dataframes(self, features, directory, test=False):
        dfs = []
        dfs_delta = []
        for sub_dir in os.listdir(directory):
            if os.path.isfile(directory + sub_dir) or sub_dir.startswith("."):
                continue
            for file in os.listdir(directory + sub_dir):
                if file.endswith(".csv"):
                    df = pd.DataFrame(pd.read_csv(directory + sub_dir + "/" + file))
                    if test:
                        df_delta = pd.DataFrame(
                            pd.read_csv(self.delta_test_path + "{}/{}".format(sub_dir, file)))
                    else:
                        df_delta = pd.DataFrame(pd.read_csv(self.delta_path + "{}/{}".format(sub_dir, file)))

                    dfs_delta.append(df_delta.reindex(columns=["weight"]))
                    dfs.append(df.reindex(columns=features))

                else:
                    continue

        return dfs, dfs_delta

    @staticmethod
    def __shuffle(x, y):
        c = list(zip(x, y))
        random.shuffle(c)
        return zip(*c)

    def __split_data(self, dfs, labels=None, scaled=False):
        if labels is None:
            x = []
            y = []
            for j in range(len(dfs)):
                df = dfs[j].copy()
                df_unscaled = dfs[j].copy()
                df[df.columns] = self.scaler.transform(df[df.columns])
                x_scaled = df.values
                x_unscaled = df_unscaled.values
                for i in range(0, len(df["weight"]) - self.days):
                    x.append(np.array(x_scaled[i:i + self.days]))
                    if scaled:
                        y.append(np.array(x_scaled[i + self.days][0]))
                    else:
                        y.append(np.array(x_unscaled[i + self.days][0]))

        else:
            x = []
            y = []
            for j in range(len(dfs)):

                df = dfs[j].copy()

                label = np.array(labels[j])

                df_unscaled = dfs[j].copy()
                df[df.columns] = self.scaler.transform(df[df.columns])
                x_scaled = df.values
                x_unscaled = df_unscaled.values

                for i in range(0, len(df["weight"]) - 30):
                    data = list(x_scaled[i:i + 30])
                    labeled_data = []
                    for d in data:
                        d = list(d)
                        d.extend(label[0])
                        labeled_data.append(list(d))

                    x.append(np.array(labeled_data))
                    if scaled:
                        y.append(np.array(x_scaled[i + 30][0]))
                    else:
                        y.append(np.array(x_unscaled[i + 30][0]))

        return x, y

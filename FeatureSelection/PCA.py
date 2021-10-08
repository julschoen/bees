import pandas as pd
import numpy as np
import os
from Clustering.StandardScaler import StandardScaler
from dataset.Dataset import Dataset


class PCA(object):

    def __init__(self):
        self.directory = "/path/to/train/"
        self.test_path = "/path/to/test/"
        self.features = ["weight", "hive_temperature", "hive_humidity", "wind_speed", "wind_speed_max",
                         "weather_temperature_min", "weather_temperature_max", "weather_temperature",
                         "weather_humidity", "rain", "rain_min", "rain_max"]

    def __load_pre_data(self):
        df = pd.DataFrame(pd.read_csv(self.directory))
        for sub_dir in os.listdir(self.directory):
            if os.path.isfile(self.directory + sub_dir) or sub_dir.startswith("."):
                continue
            for file in os.listdir(self.directory + sub_dir):
                if file.endswith(".csv"):
                    df_new = pd.DataFrame(pd.read_csv(self.directory + sub_dir + "/" + file))
                    df = df.append(pd.DataFrame(df_new), ignore_index=True)
        df.drop('datetime', axis=1, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return list(df.values)

    def __sort_eigen(self, values, vectors):
        swapped = True
        while swapped:
            swapped = False
            for i in range(len(values) - 1):
                if values[i] < values[i + 1]:
                    values, vectors = self.__swap(values, vectors, i, i + 1)
                    swapped = True
        return values, vectors

    @staticmethod
    def __swap(values, vectors, i, j):
        values = np.array(values)
        vectors = np.array(vectors)
        temp_val = values[i]
        values[i] = values[j]
        values[j] = temp_val

        temp_vec = list(vectors[:, i])
        vectors[:, i] = list(vectors[:, j])
        vectors[:, j] = list(temp_vec)

        return values, vectors

    def get_pca_dataset(self, variation):
        x = np.array(self.__load_pre_data())

        sc = StandardScaler()
        for i in range(12):
            sc.fit(x[:, i])
            x[:, i] = sc.transform(x[:, i])

        x = np.array(x)
        cov_matrix = np.cov(x.T)

        val, vec = np.linalg.eig(cov_matrix)
        val, vec = self.__sort_eigen(val, vec)

        var = sum(val)
        perc = 0

        n_components = 0
        while perc < variation:
            perc += val[n_components] / var
            n_components += 1

        dataset = Dataset()
        x, y, x_val, y_val = dataset.get_data(shuffle=True)
        x_test, y_test = dataset.get_data(test=True)

        x = np.array([np.array(j.__matmul__(vec))[:, :n_components] for j in x])
        x_val = np.array([np.array(j.__matmul__(vec))[:, :n_components] for j in x_val])
        x_test = np.array([np.array(j.__matmul__(vec))[:, :n_components] for j in x_test])

        return x, np.array(y), x_val, np.array(y_val), x_test, np.array(y_test)

    def get_n_most_important(self, n):
        x = np.array(self.__load_pre_data())
        sc = StandardScaler()
        for i in range(12):
            sc.fit(x[:, i])
            x[:, i] = sc.transform(x[:, i])

        x = np.array(x)
        cov_matrix = np.cov(x.T)
        val, vec = np.linalg.eig(cov_matrix)
        val, vec = self.__sort_eigen(val, vec)

        most_important = [np.abs(vec[:, i]).argmax() for i in range(n)]
        most_important = set(most_important)
        j = n
        while len(most_important) < n and j < len(vec):
            most_important.add(np.abs(vec[:, j]).argmax())
            j += 1

        features = list(pd.DataFrame(
            pd.read_csv(self.directory)).keys())
        features.remove('datetime')
        features.remove('Unnamed: 0')

        most_important_names = []
        for i in most_important:
            most_important_names.append(features[i])
        return most_important_names


pca = PCA()
print(pca.get_n_most_important(5))

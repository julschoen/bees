import pickle
from dataset.Dataset import Dataset

features = ['weight', 'hive_humidity', 'wind_speed_max', 'weather_temperature_min', 'rain_max']
dataset = Dataset(features=features)
algos = ['yearly', 'ending', 'delta', 'sliding_mean', 'sliding_delta_mean', 'feature_yearly',
         'feature_delta', 'cesium', 'cesium_selected', 'kshape', 'kshape_delta',
         'kshape_sliding', 'kshape_sliding_delta']

for algo in algos:
    x_train, y_train, x_val, y_val = dataset.get_cluster_data(shuffle=True, cluster_algo=algo)
    x_test, y_test = dataset.get_cluster_data(test=True, cluster_algo=algo)
    data = (x_train, y_train, x_val, y_val, x_test, y_test)
    print(algo)
    pickle.dump(data, open("{}.p".format(algo), "wb"))

from tensorflow import keras
from dataset.Dataset import Dataset
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def gru(input_shape=(22, 5)):
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(448, activation='tanh', input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.GRU(288, activation='tanh', return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.GRU(512, activation='tanh', return_sequences=False))

    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.0001))

    return model


def gru_base(input_shape=(30, 5)):
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(128, activation='tanh', input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.GRU(64, activation='tanh', return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.GRU(32, activation='tanh', return_sequences=False))

    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')

    return model


def lstm_base(input_shape=(30, 5)):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(128, activation='tanh', input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(64, activation='tanh', return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(32, activation='tanh', return_sequences=False))

    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')

    return model


def plot_test(prediction, test_data_y):
    prediction = prediction.flatten()
    plt.plot(range(len(prediction)), prediction, label="Prediction")
    plt.plot(range(len(test_data_y)), test_data_y, label="Target")
    plt.legend()
    plt.show()


print("Loading Data...")
features = ['weight', 'hive_humidity', 'wind_speed_max', 'weather_temperature_min', 'rain_max']
x_train, y_train, x_val, y_val = Dataset(features=features, days=22).get_data(shuffle=True, scaled=True)
x_test, y_test = Dataset(features=features, days=22).get_data(shuffle=False, test=True, scaled=True)

print("Making Model...")
model = gru()

monitor = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)

print("Train...")
model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), callbacks=[monitor], batch_size=16, epochs=30)

pred = model.predict(x_test)
score = model.evaluate(x_test, y_test)

score_rmse = np.sqrt(metrics.mean_squared_error(pred, y_test))
score_mae = metrics.mean_absolute_error(pred, y_test)
print("MSE: {:.10f}, RMSE: {:.10f}, MAE:  {:.10f}".format(score, score_rmse, score_mae))

plot_test(pred, y_test)

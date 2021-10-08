from tensorflow import keras
from dataset.Dataset import Dataset
import kerastuner as kt
import numpy as np


def hyper_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.GRU(units=hp.Int('units_1', min_value=32, max_value=512, step=32),
                               activation='tanh',
                               input_shape=(22, 5),
                               return_sequences=True))
    model.add(keras.layers.Dropout(hp.Float('drop_1', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(keras.layers.GRU(units=hp.Int('units_2', min_value=32, max_value=512, step=32),
                               activation='tanh',
                               return_sequences=True))
    model.add(keras.layers.Dropout(hp.Float('drop_2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(keras.layers.GRU(units=hp.Int('units_3', min_value=32, max_value=512, step=32),
                               activation='tanh',
                               return_sequences=False))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mse')
    return model


features = ['weight', 'hive_humidity', 'wind_speed_max', 'weather_temperature_min', 'rain_max']
x_train, y_train, x_val, y_val = Dataset(features=features).get_data(scaled=True, shuffle=True)

tuner_hb = kt.Hyperband(
    hyper_model,
    objective='val_loss',
    max_epochs=35,
    hyperband_iterations=3,
    directory='hypertuning'
)


monitor = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1,
                                        restore_best_weights=True)
tuner_hb.search(np.array(x_train), np.array(y_train),
                validation_data=(np.array(x_val), np.array(y_val)), batch_size=16, callbacks=[monitor])
#tuner_hb.reload()
tuner_hb.results_summary()

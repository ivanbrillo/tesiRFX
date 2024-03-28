from customLib.data_parser import load_database
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from customLib.filters import atmf, alpha_trimmed_mean_filter
from customLib.plot_drawer import plot_predictions
from customLib.plot_learning import PlotLearning


def print_model(autoencoder):
    tf.keras.utils.plot_model(autoencoder.encoder, to_file='encoder.png', show_shapes=True, show_layer_names=False)
    tf.keras.utils.plot_model(autoencoder.decoder, to_file='decoder.png', show_shapes=True, show_layer_names=False)


def get_splitted_ds(split_rate=0.8, win_size=80, alpha=40) -> tuple:
    x1, x2 = get_splitted_np(split_rate, win_size, alpha)
    return tf.data.Dataset.from_tensor_slices((x1, x1)), tf.data.Dataset.from_tensor_slices((x2, x2))


def get_splitted_np(split_rate=0.8, win_size=80, alpha=40) -> tuple:
    database, array, smooth = load_database("databse.pkl")
    database = alpha_trimmed_mean_filter(database, win_size, alpha)

    series_list = [np.array(data_dict["time_data"]) for data_dict in database]
    df = pd.DataFrame(series_list)  # each time series in a separate row
    ds = tf.data.Dataset.from_tensor_slices(df)

    len_train = int(len(ds) * split_rate)
    x_train = np.array(list(ds.take(len_train)))
    x_test = np.array(list(ds.skip(len_train)))
    x_train = x_train[:, :, np.newaxis]
    x_test = x_test[:, :, np.newaxis]

    return x_train, x_test


def mse(x, y):
    return (np.square(x - y)).mean()


def train_and_evaluate(autoencoder: Model, train_data, test_data, epochs_n=200, batch_size=100, apply_filter=False, show_latent=False, patience=200,
                       name_model="model", monitor="val_loss"):
    autoencoder.build(input_shape=(None, 1800, 1))

    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    autoencoder.compile(optimizer=optimizer, loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=False, monitor=monitor)
    checkpoint_callback = ModelCheckpoint(filepath='best_model_weights.h5', save_best_only=True, monitor=monitor, mode='min', save_weights_only=True)

    if isinstance(train_data, tf.data.Dataset):
        train = (train_data.cache().batch(batch_size),)
        train_np = np.array([y for x, y in train_data])
        test = test_data.cache().batch(batch_size)
        test_np = np.array([y for x, y in test_data])
    else:
        train = (train_data, train_data)
        train_np = train_data
        test = (test_data, test_data)
        test_np = test_data

    autoencoder.fit(
        *train,
        epochs=epochs_n,
        shuffle=True,
        validation_data=test,
        verbose=1,
        callbacks=[PlotLearning(autoencoder, show_latent=show_latent), early_stopping, checkpoint_callback],
        batch_size=batch_size
    )

    autoencoder.load_weights('best_model_weights.h5')

    decoded_values_test = np.squeeze(autoencoder.predict(test_np))
    decoded_values_train = np.squeeze(autoencoder.predict(train_np))

    if apply_filter:
        decoded_values_test = np.array([atmf(x.tolist(), 80, 40) for x in decoded_values_test])
        decoded_values_train = np.array([atmf(x.tolist(), 80, 40) for x in decoded_values_train])

    print(" ----- TEST SET ----- ")
    plot_predictions(np.squeeze(test_np), np.squeeze(decoded_values_test))

    print("\n\n\n ----- TRAIN SET ----- ")
    plot_predictions(np.squeeze(train_np), np.squeeze(decoded_values_train))

    print("\n\n\nTEST SET MSE:", mse(np.squeeze(test_np), np.squeeze(decoded_values_test)))
    print("TRAINING SET MSE:", mse(np.squeeze(train_np), np.squeeze(decoded_values_train)))
    autoencoder.save_weights(name_model + ".h5")

from tesiRFX.tesiRFX.data_parser import create_db
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output
import math
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


class PlotLearning(Callback):

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        metrics = [x for x in logs if 'val' not in x]
        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        plt.ylim(0, 0.05)
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs.plot(range(1, epoch + 2),
                     self.metrics[metric],
                     label=metric)
            if logs['val_' + metric]:
                axs.plot(range(1, epoch + 2),
                         self.metrics['val_' + metric],
                         label='val_' + metric)

            axs.legend()
            axs.grid()

        plt.tight_layout()
        plt.show()


def atmf(data: np.array, win_size: int, alpha: int) -> list:
    n = len(data)
    result = [0] * (n - win_size)

    assert win_size > alpha > 0 and alpha % 4 == 0 and win_size % 2 == 0

    for i in range(win_size // 2, n - win_size // 2):
        window = data[i - win_size // 2:i + win_size // 2]
        window.sort()
        result[i - win_size // 2] = sum(window[alpha // 4:win_size - (alpha // 4) * 3]) / (win_size - alpha)

    return [result[0]] * (win_size // 2) + result + [result[-1]] * (win_size // 2)


def get_splitted_ds(split_rate=0.8, win_size=80, alpha=40) -> tuple:
    x1, x2 = get_splitted_np(split_rate, win_size, alpha)
    return tf.data.Dataset.from_tensor_slices((x1, x1)), tf.data.Dataset.from_tensor_slices((x2, x2))


def get_splitted_np(split_rate=0.8, win_size=80, alpha=40) -> tuple:
    database = create_db("tesiRFX/tesiRFX/data")
    series_list = [data_dict["time_data"].to_numpy() for data_dict in database]
    df = pd.DataFrame(series_list)  # each time series in a separate row
    ds = tf.data.Dataset.from_tensor_slices(df)

    len_train = int(len(ds) * split_rate)
    x_train = np.array(list(ds.take(len_train)))
    x_test = np.array(list(ds.skip(len_train)))

    x_train_smoothed = np.array([atmf(x.tolist(), win_size, alpha) for x in x_train])
    x_test_smoothed = np.array([atmf(x.tolist(), win_size, alpha) for x in x_test])

    return x_train_smoothed, x_test_smoothed


def grid_plot(original_data: np.array, decoded_data: np.array) -> None:
    n = len(original_data)
    fig, axes = plt.subplots(ncols=4, nrows=math.ceil(n / 4), layout='constrained', figsize=(3.5 * 4, 3.5 * math.ceil(n / 4)))

    for index in range(n):
        axes[int(index / 4)][index % 4].plot(original_data[index])
        axes[int(index / 4)][index % 4].plot(decoded_data[index])

    plt.show()


def MSE(x, y):
    return (np.square(x - y)).mean()


def train_and_evaluate(autoencoder: Model, train_data, test_data, use_callback=True, epochs_n=200, batch_size=50, apply_filter=False):
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

    checkpoint_callback = ModelCheckpoint(
        filepath='best_model_weights.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
    )

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
        callbacks=[PlotLearning(), early_stopping, checkpoint_callback] if use_callback else []
    )

    decoded_values_test = np.squeeze(np.array(autoencoder.call(test_np)))
    decoded_values_train = np.squeeze(np.array(autoencoder.call(train_np)))

    if apply_filter:
        decoded_values_test = np.array([atmf(x.tolist(), 80, 40) for x in decoded_values_test])
        decoded_values_train = np.array([atmf(x.tolist(), 80, 40) for x in decoded_values_train])

    print(" ----- TEST SET ----- ")
    grid_plot(np.squeeze(test_np), np.squeeze(decoded_values_test))

    print("\n\n\n ----- TRAIN SET ----- ")
    grid_plot(np.squeeze(train_np), np.squeeze(decoded_values_train))

    print("\n\n\nTEST SET MSE:", MSE(np.squeeze(test_np), np.squeeze(decoded_values_test)))
    print("TRAINING SET MSE:", MSE(np.squeeze(train_np), np.squeeze(decoded_values_train)))

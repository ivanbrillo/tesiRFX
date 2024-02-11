import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(1800, 1)),
            layers.Conv1D(filters=18, kernel_size=75, activation='relu', strides=4, padding="same"),
            layers.Conv1D(filters=28, kernel_size=30, activation='relu', strides=4, padding="same"),
            layers.Flatten(),
            layers.Dense(140, activation='relu'),
            layers.Dense(120, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(80, activation='linear'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(80,)),
            layers.Dense(100, activation='linear'),
            layers.Dense(120, activation='relu'),
            layers.Dense(140, activation='relu'),
            layers.Dense(113 * 28, activation='relu'),
            layers.Reshape((113, 28)),
            layers.Conv1DTranspose(filters=28, kernel_size=30, activation='relu', strides=4, padding="same", output_padding=2),
            layers.Conv1DTranspose(filters=18, kernel_size=75, activation='relu', strides=4, padding="same"),
            layers.Conv1DTranspose(filters=1, kernel_size=81, activation='linear', padding="same"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class FullAutoencoder(Model):
    def __init__(self, autoencoder):
        super(FullAutoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(1800, 1)),
            autoencoder.encoder,
            layers.Dense(70, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(4, activation='linear'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(4,)),
            layers.Dense(10, activation='linear'),
            layers.Dense(30, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(70, activation='relu'),
            layers.Dense(80, activation='linear'),
            autoencoder.decoder,
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

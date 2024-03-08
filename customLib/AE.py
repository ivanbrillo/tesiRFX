import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class AE(Model):
    def __init__(self, encoder: tf.keras.Sequential, decoder: tf.keras.Sequential):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def set_trainable(self, trainable):
        self.trainable = trainable
        self.encoder.trainable = trainable
        self.decoder.trainable = trainable


def get_sequentials_outer():
    encoder = tf.keras.Sequential([
        layers.Input(shape=(1800, 1)),
        layers.Conv1D(filters=18, kernel_size=75, activation='relu', strides=4, padding="same"),
        layers.Conv1D(filters=28, kernel_size=30, activation='relu', strides=4, padding="same"),
        layers.Flatten(),
        layers.Dense(140, activation='relu'),
        layers.Dense(120, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(80, activation='linear'),
    ])

    decoder = tf.keras.Sequential([
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

    return encoder, decoder


def get_seq_full(autoencoder: AE, latent_dim: int):
    encoder = tf.keras.Sequential([
        layers.Input(shape=(1800, 1)),
        autoencoder.encoder,
        layers.Dense(70, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(30, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(latent_dim, activation='linear'),
    ])

    decoder = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(10, activation='linear'),
        layers.Dense(30, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(70, activation='relu'),
        layers.Dense(80, activation='linear'),
        autoencoder.decoder,
    ])

    return encoder, decoder

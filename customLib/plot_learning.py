from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class PlotLearning(Callback):
    def __init__(self, autoencoder=None, data=None, show_latent=False):
        super().__init__()
        self.autoencoder = autoencoder
        self.data = data
        self.show_latent = show_latent
        self.metrics = {}

    def on_train_begin(self, logs=None):
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs=None):
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        metrics = [x for x in logs if 'val' not in x]
        new_plt = 1 if self.show_latent else 0
        f, axs = plt.subplots(1, len(metrics) + new_plt, figsize=(15, 5))

        if not isinstance(axs, np.ndarray):
            axs = (axs,)
            plt.ylim(0, 0.05)
        else:
            for ax in axs[:-new_plt]:
                ax.set_ylim((0, 10))

        pred = None
        if self.show_latent:
            pred = self.autoencoder.encoder.predict(self.data)[2].T

        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        if self.show_latent:
            axs[len(metrics)].scatter(pred[0], pred[1])
            axs[len(metrics)].grid()

        plt.tight_layout()
        plt.show()

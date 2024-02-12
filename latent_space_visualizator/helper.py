import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import models

names = {
    "exposition time": "exposure time",
    "DBD vs PT": "type",
    "storage time": "supply delay",
    "power or freq": "frequency",
}


def load_autoencoder(dim: int) -> models.FullAutoencoder:
    full_autoencoder = models.FullAutoencoder(models.Autoencoder(), dim)
    full_autoencoder.build(input_shape=(None, 1800, 1))
    full_autoencoder.load_weights(f"./weights/FullCOnvAE{str(dim)}Dbis.h5")
    return full_autoencoder


def load_database(path: str) -> tuple[dict, np.array]:
    with open(path, 'rb') as f:
        database = pickle.load(f)

    all_np_array = [np.reshape(i["time_data"].values, [1, 1800, 1]) for i in database]
    return database, np.concatenate(all_np_array, axis=0)


def setup_frame(title: str) -> tuple[Figure, list]:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    plt.subplots_adjust(left=0.3)
    fig.canvas.manager.set_window_title(title)
    fig.suptitle('Latent Space Visualization', fontsize=16)
    ax[0].set(xlabel='first dimension', ylabel='second dimension')
    ax[1].set(xlabel='third dimension', ylabel='fourth dimension')
    return fig, ax


def create_colors(label: str, database: dict, ax: list) -> tuple[dict, list]:
    unique_values = {i[label] for i in database}

    color_palette = plt.cm.get_cmap('tab10', len(unique_values))
    colors_dict = {element: color_palette(i) for (i, element) in enumerate(unique_values)}
    legend_handles = [plt.scatter([], [], color=colors_dict[i], label=i) for i in colors_dict]

    ax[0].legend(handles=legend_handles)
    ax[1].legend(handles=legend_handles)

    return colors_dict, legend_handles


def get_colors_list(database: dict, colors: dict, label: str) -> list:
    return [colors[i[label]] for i in database]


def scale(ax, values):
    x_min, x_max = min(values[0]), max(values[0])
    y_min, y_max = min(values[1]), max(values[1])

    ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

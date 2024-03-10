import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from customLib.AE import *

names = {
    "exposition time": "exposure time",
    "DBD vs PT": "type",
    "storage time": "supply delay",
    "power or freq": "frequency",
    "date": "date"
}


def load_autoencoder(dim: int) -> AE:
    outer = AE(*get_sequentials_outer())
    outer.set_trainable(False)
    full_autoencoder = AE(*get_seq_full(outer, dim))
    full_autoencoder.build(input_shape=(None, 1800, 1))
    full_autoencoder.load_weights(f"../weights/FullConvAE{str(dim)}Dbis.h5")
    return full_autoencoder


def setup_frame(title: str) -> tuple[Figure, list]:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    plt.subplots_adjust(left=0.3)
    fig.canvas.manager.set_window_title(title)
    fig.suptitle('Latent Space Visualization', fontsize=16)
    ax[0].set(xlabel='first dimension', ylabel='second dimension')
    ax[1].set(xlabel='third dimension', ylabel='fourth dimension')
    return fig, ax


def create_colors(label: str, database: list, ax: list) -> tuple[dict, list]:
    unique_values = {i[label] for i in database}

    color_palette = plt.cm.get_cmap('tab10', len(unique_values))
    colors_dict = {element: color_palette(i) for (i, element) in enumerate(unique_values)}

    if "0d" in colors_dict.keys() and "100s" in colors_dict.keys():
        colors_dict["0d"] = colors_dict["100s"]

    if "physical" in unique_values:
        colors_dict["physical"] = (0.0, 0.0, 0.0, 0.4)

    legend_handles = [plt.scatter([], [], color=colors_dict[i], label=i) for i in colors_dict]

    ax[0].legend(handles=legend_handles)
    ax[1].legend(handles=legend_handles)

    return colors_dict, legend_handles


def get_colors_list(database: list, colors: dict, label: str) -> list:
    return [colors[i[label]] for i in database]


def scale(ax, values):
    x_min, x_max = min(values[0]), max(values[0])
    y_min, y_max = min(values[1]), max(values[1])

    ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))


def _interpolate_series(series_list: list) -> np.array:
    interpolate_list = list()

    x_interpolated = np.linspace(0, len(series_list[0]) - 1, 1800)
    x_original = np.arange(len(series_list[0]))

    for data in series_list:
        interpolated_function = interp1d(x_original, data, kind='linear')
        interpolate_list.append(interpolated_function(x_interpolated))

    return np.array(interpolate_list)

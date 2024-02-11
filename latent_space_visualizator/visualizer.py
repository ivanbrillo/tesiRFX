import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import RadioButtons
import models
from tensorflow.keras.models import Model

names = {
    "exposition time": "exposure time",
    "DBD vs PT": "type",
    "storage time": "supply delay",
    "power or freq": "frequency",
}


def load_autoencoder(weights: str) -> Model:
    full_autoencoder = models.FullAutoencoder(models.Autoencoder())
    full_autoencoder.build(input_shape=(None, 1800, 1))
    full_autoencoder.load_weights(weights)
    return full_autoencoder


def load_database(path: str) -> tuple:
    with open(path, 'rb') as f:
        database = pickle.load(f)

    all_np_array = [np.reshape(i["time_data"].values, [1, 1800, 1]) for i in database]
    return database, np.concatenate(all_np_array, axis=0)


def setup_frame(title: str) -> tuple[Figure, list]:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    plt.subplots_adjust(left=0.3)
    fig.canvas.manager.set_window_title(title)
    fig.suptitle('Latent Space Visualization', fontsize=16)
    ax[0].set_xlabel('first dimension')
    ax[0].set_ylabel('second dimension')
    ax[1].set_xlabel('third dimension')
    ax[1].set_ylabel('fourth dimension')
    return fig, ax


def create_colors(label: str, database: dict, ax: list) -> tuple:
    unique_values = set()
    for i in database:
        unique_values.add(i[label])

    color_palette = plt.cm.get_cmap('tab10', len(unique_values))
    colors_dict, legend_handles = ({}, [])

    for i, element in enumerate(unique_values):
        colors_dict[element] = color_palette(i)

    for i in colors_dict:
        legend_handles.append(plt.scatter([], [], color=colors_dict[i], label=i))

    ax[0].legend(handles=legend_handles)
    ax[1].legend(handles=legend_handles)

    return colors_dict, legend_handles


def get_colors_list(database: dict, colors: dict, label: str) -> list:
    return [colors[i[label]] for i in database]


class Visualizer:

    def __init__(self):
        self.fig, self.ax = setup_frame("Calcium Signals' Latent Space Visualizer")
        self.cid1 = self.ax[0].figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = self.ax[1].figure.canvas.mpl_connect('button_press_event', self.onclick)

        self.full_autoencoder = load_autoencoder("./weights/prova2.h5")
        self.database, np_arrays = load_database('databse.pkl')
        self.x = self.full_autoencoder.encoder.predict(np_arrays, verbose=0).T

        self.current_label = "exposure time"
        self.selected = [False, False]
        self.coordinate = [0] * 4
        self.scatter = []
        self.scatter_plot()
        self.radio_button = self.create_radiobutton()

    def create_radiobutton(self) -> RadioButtons:
        rax = plt.axes([0.05, 0.4, 0.15, 0.30])
        radio = RadioButtons(rax, list(names.keys()), 0)
        radio.on_clicked(self.change_type)
        return radio

    def scatter_plot(self):
        colors, self.scatter = create_colors(self.current_label, self.database, self.ax)
        all_colors = get_colors_list(self.database, colors, self.current_label)

        self.scatter.append(self.ax[0].scatter(self.x[0], self.x[1], color=all_colors))
        self.scatter.append(self.ax[1].scatter(self.x[2], self.x[3], color=all_colors))

    def change_type(self, new_label: str):
        self.current_label = names[new_label]

        for s in self.scatter:
            s.remove()

        self.fig.canvas.draw_idle()
        self.scatter_plot()

    def onclick(self, event):
        if event.inaxes is self.ax[0]:
            self.coordinate[0] = event.xdata
            self.coordinate[1] = event.ydata
            self.selected[0] = True
        elif event.inaxes is self.ax[1]:
            self.coordinate[2] = event.xdata
            self.coordinate[3] = event.ydata
            self.selected[1] = True
        self.generate_plt()

    def generate_plt(self):
        if self.selected == [True, True]:
            fig_new, ax_new = plt.subplots(figsize=(6, 6))
            coordinate_np = np.array(self.coordinate).reshape([1, 4])
            values = self.full_autoencoder.decoder(coordinate_np)

            ax_new.plot(values[0])
            ax_new.set_title('Generated Time Series')
            plt.show()

            self.selected = [False, False]

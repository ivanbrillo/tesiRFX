import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import helper


class Visualizer:

    def __init__(self):
        self.fig, self.ax = helper.setup_frame("Calcium Signals' Latent Space Visualizer")
        self.cid1 = self.ax[0].figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid3 = self.ax[0].figure.canvas.mpl_connect('pick_event', self.onpick)
        self.cid2 = self.ax[1].figure.canvas.mpl_connect('button_press_event', self.onclick)

        self.current_dim = 2
        self.full_autoencoder = helper.load_autoencoder(self.current_dim)
        self.database, self.np_arrays, self.np_smooth = helper.load_database('databse.pkl')
        self.x = np.array(0)
        self.generate_values()

        self.current_label = "exposure time"
        self.selected = [False] * (self.current_dim // 2)
        self.coordinate = [0] * self.current_dim
        self.scatter = []
        self.scatter_plot()
        self.radio_button = self.create_radiobutton()
        self.picked = (False, 0)

    def generate_values(self):
        self.x = self.full_autoencoder.encoder.predict(self.np_smooth, verbose=0).T

    def create_radiobutton(self) -> tuple[RadioButtons, RadioButtons]:
        rax = plt.axes([0.05, 0.3, 0.15, 0.2])
        radio1 = RadioButtons(rax, list(helper.names.keys()), 0)

        rax = plt.axes([0.05, 0.5, 0.15, 0.12])
        radio2 = RadioButtons(rax, ["2D", "4D"], 0)

        radio1.on_clicked(self.change_type)
        radio2.on_clicked(self.change_dim)
        return radio1, radio2

    def scatter_plot(self):
        colors, self.scatter = helper.create_colors(self.current_label, self.database, self.ax)
        all_colors = helper.get_colors_list(self.database, colors, self.current_label)

        self.scatter.append(self.ax[0].scatter(self.x[0], self.x[1], color=all_colors, picker=True))
        helper.scale(self.ax[0], self.x[:2, :])

        if self.current_dim == 4:
            self.scatter.append(self.ax[1].scatter(self.x[2], self.x[3], color=all_colors, picker=True))
            helper.scale(self.ax[1], self.x[2:, :])

    def change_type(self, new_label: str):
        self.current_label = helper.names[new_label]
        self.clear_and_plot()

    def clear_and_plot(self):
        for s in self.scatter:
            s.remove()

        self.fig.canvas.draw_idle()
        self.scatter_plot()

    def change_dim(self, label: str):
        self.current_dim = int(label[0])
        self.full_autoencoder = helper.load_autoencoder(self.current_dim)
        self.generate_values()
        self.clear_and_plot()

        self.selected = [False] * (self.current_dim // 2)
        self.coordinate = [0] * self.current_dim

    def onclick(self, event):
        if event.inaxes is self.ax[0]:
            self.coordinate[0] = event.xdata
            self.coordinate[1] = event.ydata
            self.selected[0] = True
        elif event.inaxes is self.ax[1] and self.current_dim == 4:
            self.coordinate[2] = event.xdata
            self.coordinate[3] = event.ydata
            self.selected[1] = True
        self.generate_plt()

    def onpick(self, event):
        if self.current_dim == 2:
            self.picked = (True, event.ind[0])

    def generate_plt(self):
        if all(self.selected):
            fig_new, ax_new = plt.subplots(figsize=(6, 6))
            coordinate_np = np.array(self.coordinate).reshape([1, self.current_dim])

            if self.picked[0]:
                coordinate_np = np.array([self.x[0][self.picked[1]], self.x[1][self.picked[1]]]).reshape([1, self.current_dim])

            values = self.full_autoencoder.decoder(coordinate_np)

            ax_new.plot(values[0], label='Reconstructed')

            if self.picked[0]:
                coordinate_np1 = self.np_arrays[self.picked[1]].reshape(1800, )
                coordinate_np2 = self.np_smooth[self.picked[1]].reshape(1800, )

                ax_new.plot(coordinate_np1, label='Original')
                ax_new.plot(coordinate_np2, label='Smoothed')

            ax_new.set_title('Generated Time Series')
            ax_new.legend()
            plt.show()

            self.picked = (False, 0)
            self.selected = [False] * (self.current_dim // 2)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import models
import pickle

plt.switch_backend('TkAgg')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig.canvas.manager.set_window_title("Calcium Signals' Latent Space Visualizer")  # Set the title of the Tkinter window

full_autoencoder = models.FullAutoencoder(models.Autoencoder())
full_autoencoder.build(input_shape=(None, 1800, 1))
full_autoencoder.load_weights("./weights/prova2.h5")

# database = data_parser.create_db("./data")
# with open('databse.pkl', 'wb') as f:
#     pickle.dump(database, f)

with open('databse.pkl', 'rb') as f:
    database = pickle.load(f)

colors = {}
legend_handles = []

names = {
    "exposition time": "exposure time",
    "DBD vs PT": "type",
    "storage time": "supply delay",
    "power or freq": "frequency",
}


def create_colors(label: str):
    global colors, legend_handles
    time_set = set()

    for i in database:
        time_set.add(i[names[label]])

    color_palette = plt.cm.get_cmap('tab10', len(time_set))
    colors = {}
    legend_handles = []

    for i, element in enumerate(time_set):
        colors[element] = color_palette(i)

    for i in colors:
        legend_handles.append(plt.scatter([], [], color=colors[i], label=i))
    ax[0].legend(handles=legend_handles)
    ax[1].legend(handles=legend_handles)


create_colors("exposition time")
all_np_arrays = []
all_colors = []
all_lables = []

for i in database:
    np_array = i["time_data"].values
    np_array = np.reshape(np_array, [1, 1800, 1])
    all_np_arrays.append(np_array)
    all_colors.append(colors[i["exposure time"]])
    all_lables.append(i["exposure time"])

# Concatenate all arrays into a single array
all_np_arrays = np.concatenate(all_np_arrays, axis=0)

# Compute predictions for all arrays at once
x1, x2, x3, x4 = full_autoencoder.encoder.predict(all_np_arrays, verbose=0).T

plt.subplots_adjust(left=0.3)

# Plot the latent space for each sample
scatter1 = ax[0].scatter(x1, x2, color=all_colors)
ax[0].set_xlabel('first dimension')
ax[0].set_ylabel('second dimension')

scatter2 = ax[1].scatter(x3, x4, color=all_colors, label=all_lables)
ax[1].set_xlabel('third dimension')
ax[1].set_ylabel('fourth dimension')

coordinate = [0] * 4
set1 = False
set2 = False

rax = plt.axes([0.05, 0.4, 0.15, 0.30])
radio = RadioButtons(rax, ['exposition time', 'DBD vs PT', 'storage time', "power or freq"], 0, activecolor='g')


def change_type(label):
    global all_colors, scatter1, scatter2
    create_colors(label)
    all_colors = []
    for i in database:
        all_colors.append(colors[i[names[label]]])

    scatter1.remove()
    scatter2.remove()

    scatter1 = ax[0].scatter(x1, x2, color=all_colors)
    scatter2 = ax[1].scatter(x3, x4, color=all_colors)

    for e in legend_handles:
        e.remove()

    fig.canvas.draw_idle()


radio.on_clicked(change_type)


def generate_plt():
    global set1, set2
    if set1 and set2:
        fig_new, ax_new = plt.subplots(figsize=(6, 6))
        coordinate_np = np.array(coordinate).reshape([1, 4])
        values = full_autoencoder.decoder(coordinate_np)

        ax_new.plot(values[0])
        ax_new.set_title('Generated Time Series')
        plt.show()

        set1 = False
        set2 = False


def onclick(event):
    global set1, set2
    if event.inaxes is ax[0]:
        coordinate[0] = event.xdata
        coordinate[1] = event.ydata
        set1 = True
    if event.inaxes is ax[1]:
        coordinate[2] = event.xdata
        coordinate[3] = event.ydata
        set2 = True
    generate_plt()


cid1 = ax[0].figure.canvas.mpl_connect('button_press_event', onclick)
cid2 = ax[1].figure.canvas.mpl_connect('button_press_event', onclick)

fig.suptitle('Latent Space Visualization', fontsize=16)
plt.show()

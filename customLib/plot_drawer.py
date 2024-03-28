import matplotlib.pyplot as plt
import math
import numpy as np


def subplot_grid(data_list: list, axes: list, legend: bool):
    for index, series in enumerate(data_list):
        legend_value = None
        if legend:
            legend_str = f"{series['date']} {series['type']} {series['frequency']} {series['exposure time']} {series['supply delay']}"
            legend_value = legend_str if legend is None else f"{legend_str} {legend}"

        axes[int(index / 4)][index % 4].plot(series["time_data"].index.tolist(), series["time_data"].values, label=legend_value)
        axes[int(index / 4)][index % 4].legend(loc='lower right', fontsize=7)


def subplot(data_list: list, axes: list, legend: str = None):
    for ax, series in zip(axes, data_list):
        legend_str = f"{series['date']} {series['type']} {series['frequency']} {series['exposure time']} {series['supply delay']}"
        legend_value = legend_str if legend is None else f"{legend_str} {legend}"

        ax.plot(series["time_data"].index.tolist(), series["time_data"].values, label=legend_value)
        ax.legend(loc='lower right', fontsize=7)


def plot(original_data: list, modified_data: list = None, modified_legend: str = None) -> None:
    #  1xN grid of subplot
    fig, axes = plt.subplots(1, len(original_data), figsize=(5 * len(original_data), 5))
    subplot(original_data, axes)

    if modified_data is not None:
        subplot(modified_data, axes, modified_legend)

    plt.show()


def grid_plot(original_data, new_data=None, legend=False) -> None:
    n = len(original_data)  # (N/4)x4 grid of subplot
    fig, axes = plt.subplots(ncols=4, nrows=math.ceil(n / 4), layout='constrained', figsize=(3.5 * 4, 3.5 * math.ceil(n / 4)))
    subplot_grid(original_data, axes, legend)

    if new_data is not None:
        subplot_grid(new_data, axes, legend)

    plt.show()


def plot_predictions(original_data: np.array, decoded_data: np.array, legend1=None, legend2=None, unit=None) -> None:
    n = len(original_data)
    fig, axes = plt.subplots(ncols=4, nrows=math.ceil(n / 4), layout='constrained', figsize=(3.5 * 4, 3.5 * math.ceil(n / 4)))

    for index in range(n):
        axes[int(index / 4)][index % 4].plot(original_data[index])

        if unit is not None:
            axes[int(index / 4)][index % 4].set_xlabel('Numero campionamento')
            axes[int(index / 4)][index % 4].set_ylabel('Concentrazione calcio citosol (µM)')

        if decoded_data is not None:
            axes[int(index / 4)][index % 4].plot(decoded_data[index])

        if legend1 is not None and legend2 is not None:
            axes[int(index / 4)][index % 4].legend([legend1, legend2], fontsize=7)
        elif legend1 is not None:
            axes[int(index / 4)][index % 4].legend([legend1], fontsize=7)

    plt.show()

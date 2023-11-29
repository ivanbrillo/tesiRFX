import matplotlib.pyplot as plt
import math


def subplot(data_list: list, axes: list, legend: str = None):
    for index, series in enumerate(data_list):
        legend_str = f"{series['date']} {series['type']} {series['frequency']} {series['exposure time']} {series['supply delay']}"
        legend_value = legend_str if legend is None else f"{legend_str} {legend}"

        axes[int(index / 4)][index % 4].plot(series["time_data"].index.tolist(), series["time_data"].values,
                                             label=legend_value)
        axes[int(index / 4)][index % 4].legend(loc='lower right', fontsize=7)


def plot(original_data: list, modified_data: list = None, modified_legend: str = None) -> None:
    #  1xN grid of subplot
    fig, axes = plt.subplots(1, len(original_data), figsize=(5 * len(original_data), 5))
    subplot(original_data, axes)

    if modified_data is not None:
        subplot(modified_data, axes, modified_legend)

    plt.show()


def grid_plot(original_data: list) -> None:
    #  (N/4)x4 grid of subplot
    N = len(original_data)
    fig, axes = plt.subplots(ncols=4, nrows=math.ceil(N / 4), layout='constrained',
                             figsize=(3.5 * 4, 3.5 * math.ceil(N / 4)))

    subplot(original_data, axes)
    # plt.tight_layout()
    plt.show()

from matplotlib import pyplot as plt


def scatter_plot_latent(ax, encoded_values, all_colors=None):
    if all_colors is None:
        all_colors = []

    ax.scatter(encoded_values[0], encoded_values[1], color=all_colors)
    # scale(ax, encoded_values)


def scale(ax, values):
    x_min, x_max = min(values[0]), max(values[0])
    y_min, y_max = min(values[1]), max(values[1])

    ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))


def create_colors(label: str, database: list, ax=None) -> tuple[dict, list]:
    unique_values = {i[label] for i in database}

    color_palette = plt.cm.get_cmap('tab10', len(unique_values))
    colors_dict = {element: color_palette(i) for (i, element) in enumerate(unique_values)}

    if "0d" in colors_dict.keys() and "100s" in colors_dict.keys():
        colors_dict["0d"] = colors_dict["100s"]

    if "physical" in unique_values:
        colors_dict["physical"] = (0.0, 0.0, 0.0, 0.4)

    legend_handles = None
    if ax is not None:
        legend_handles = [plt.scatter([], [], color=colors_dict[i], label=i) for i in colors_dict]
        ax[0].legend(handles=legend_handles)
        ax[1].legend(handles=legend_handles)

    return colors_dict, legend_handles


def get_colors_list(database: list, colors: dict, label: str) -> list:
    return [colors[i[label]] for i in database]

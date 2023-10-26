def subplot(dataframe: pd.DataFrame, axes: list, legend: str = None):
    column_names = dataframe.columns.values.tolist()
    del column_names[0]  # remove TIME column

    for index, name in enumerate(column_names):
        legend_value = name if legend is None else f"{legend} {name}"
        axes[index].plot(dataframe["TIME"], dataframe[name], label=legend_value)
        axes[index].legend(loc='lower right')


def plot(original_data: pd.DataFrame, modified_data: pd.DataFrame = None, modified_legend: str = None) -> None:
    #  1xN grid of subplot
    fig, axes = plt.subplots(1, len(column_names), figsize=(15, 3))

    subplot(original_data, axes)

    if modified_data is not None:
        subplot(modified_data, axes, modified_legend)

    plt.show()

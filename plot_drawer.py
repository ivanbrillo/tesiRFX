import matplotlib.pyplot as plt
import pandas as pd

def get_columns(dataframe: pd.DataFrame) -> list:
    column_names = dataframe.columns.values.tolist()
    del column_names[0]  # remove TIME column
    return column_names


def subplot(dataframe: pd.DataFrame, axes: list, legend: str = None):
    for index, name in enumerate(get_columns(dataframe)):
        legend_value = name if legend is None else f"{legend} {name}"
        axes[index].plot(dataframe["TIME"], dataframe[name], label=legend_value)
        axes[index].legend(loc='lower right')


def plot(original_data: pd.DataFrame, modified_data: list = None, modified_legend: str = None) -> None:
    #  1xN grid of subplot
    fig, axes = plt.subplots(1, len(get_columns(original_data)), figsize=(15, 3))
    subplot(original_data, axes)

    if modified_data is not None:
      for index, frame in enumerate(modified_data):
        subplot(frame, axes, f'({index}) {modified_legend}' )

    plt.show()

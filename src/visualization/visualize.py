"""Visualization."""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")


def plot_data(
    x_data, y_data, x_label: str, y_label: str, title: str, path: str
) -> None:
    """Plot data using to matplotlib.pylot.plot.

    Args:
        x (_type_): any array
        y (_type_): any array
        x_label (str): X label
        y_label (str): Y label
        title (str): Title
        path (str): Path to save the plot

    Raises
        Exception
    """
    try:
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if path is not None:
            plt.savefig(path)

    except Exception as exception:
        raise Exception from exception

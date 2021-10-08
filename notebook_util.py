from typing import Tuple
from collections import namedtuple
import numpy as np

from matplotlib import pyplot as plt

TDLearningRes = namedtuple(
    "TDLearningRes", ["alpha", "gamma", "mean_reward", "mean_step", "test_mean_rewards"])


def make_grid(x_axis_values, y_axis_values, x_values, y_values, z_values, fill_value: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(x_axis_values) >= len(
        set(x_values)), "Number of measurements must be greather than number of the 'x_values'"
    assert x_values.min() >= x_axis_values.min() and x_values.max() <= x_axis_values.max()
    assert y_values.min() >= y_axis_values.min() and y_values.max() <= y_axis_values.max()
    assert len(y_axis_values) >= len(
        set(y_values)), "Number of measurements must be greather than number of the 'y_values'"

    x, y = np.meshgrid(x_axis_values, y_axis_values, indexing="xy")

    z_grid = np.full((len(y_axis_values), len(x_axis_values)),
                     fill_value=fill_value, dtype=z_values.dtype)

    x_indices = np.digitize(x_values, x_axis_values, right=True)
    y_indices = np.digitize(y_values, y_axis_values, right=True)

    z_grid[y_indices, x_indices] = z_values

    return x, y, z_grid


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    if cbar_kw is None:
        cbar_kw = dict()

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(labels=col_labels)
    ax.set_yticklabels(labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

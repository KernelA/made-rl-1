from typing import Tuple, List
from collections import namedtuple
from itertools import product

from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
import gym

from rl_learning import QTableDict, EpsilonGreedyPolicy, QLearningSimulation


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


def plot_v_function(data: pd.DataFrame, title: str):
    fig = plt.figure()
    ax = fig.add_subplot()

    player_axis_start = 2
    dealer_axis_start = 1
    player_sum = np.arange(player_axis_start, 21 + 2)
    dealer_show = np.arange(dealer_axis_start, 10 + 2)

    _, _, rewards_data = make_grid(player_sum, dealer_show, data["player_sum"].to_numpy(
    ), data["dealer_open_card"].to_numpy(), data["reward"].to_numpy())

    heatmap(rewards_data, row_labels=list(map(str, dealer_show)),
            col_labels=list(map(str, player_sum)), ax=ax, cmap="rainbow")

    ax.set_xlabel("Сумма карт игрока")
    ax.set_ylabel("Открытая карта диллера")
    ax.set_title(title)

    return fig


def generate_stat(td_learning_cls, gammas: np.ndarray, alpha: np.ndarray,
                  total_episodes: int, total_test_episodes: int, q_function, policy, is_learning: bool) -> List[TDLearningRes]:
    values = []
    for gamma_param, alpha in product(gammas, alpha):
        td_learning = td_learning_cls(gym.make(
            "Blackjack-v0"), policy, q_function, alpha=alpha, gamma=gamma_param, is_learning=is_learning)
        td_res = td_learning.simulate(total_episodes, total_test_episodes)
        values.append(TDLearningRes(alpha, gamma_param, td_res.mean_reward,
                                    td_res.mean_step, td_res.test_mean_rewards))
    return values


def plot_td_learning_stat(alpha_axis: np.ndarray, gamma_axis: np.ndarray, results: List[TDLearningRes], title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    values = np.array(list(map(lambda x: (x.alpha, x.gamma, x.mean_reward), results)))

    alphas = values[:, 0]
    gammas = values[:, 1]
    mean_rewards = values[:, 2]

    _, _, mean_reward = make_grid(alpha_axis, gamma_axis, alphas, gammas, mean_rewards)

    heatmap(mean_reward, row_labels=list(
        map(lambda x: f"{x:.2f}", gamma_axis)), col_labels=list(map(str, alpha_axis)), ax=ax)

    ax.set_xlabel("Скорость обучения alpha")
    ax.set_ylabel("gamma")
    ax.set_title(title)

    return fig


def plot_training_process(rewards: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    _, trend = sm.tsa.filters.hpfilter(rewards)
    episodes = tuple(range(len(rewards)))
    ax.plot(episodes, rewards)
    ax.plot(episodes, trend, label="Тренд", c="red")
    ax.set_title("Средняя награда во время обучения")
    ax.set_xlabel("Номер эпизода")
    ax.set_ylabel("Средняя награда")
    plt.legend()

    return fig


def plot_stat(alpha_values: np.ndarray, gamma_values: np.ndarray, train_stat: List[TDLearningRes],
              test_stat: List[TDLearningRes]):
    train_fig = plot_td_learning_stat(alpha_values, gamma_values,
                                      train_stat, "Средняя награда во время обучения")

    best_stat = max(train_stat, key=lambda x: x.mean_reward)

    training_process_fig = plot_training_process(best_stat.test_mean_rewards)

    test_fig = plot_td_learning_stat(
        alpha_values, gamma_values, test_stat, "Средняя награда при использовании обученной стратегии")

    return train_fig, training_process_fig, test_fig


def evaluate_td_learning(alpha_values: np.ndarray, gamma_values: np.ndarray, epsilon: float,
                         num_train_episodes: int, num_test_episodes: int, learning_cls=QLearningSimulation):
    q_function = QTableDict()

    policy = EpsilonGreedyPolicy(q_function, epsilon=epsilon)

    train_stat = generate_stat(learning_cls, gamma_values, alpha_values,
                               num_train_episodes, num_test_episodes, q_function, policy, is_learning=True)

    policy = EpsilonGreedyPolicy(q_function, epsilon=0)

    test_stat = generate_stat(learning_cls, gamma_values, alpha_values,
                              num_train_episodes, None, q_function, policy, is_learning=False)

    return train_stat, test_stat, q_function

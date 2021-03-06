from typing import Iterable, Tuple, List, Optional
from collections import namedtuple
from itertools import product, repeat
from multiprocessing import Pool
import copy
import logging

from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd


from rl_learning import QTableDict, EpsilonGreedyPolicy, QLearningSimulation
from rl_learning.utils import Action, get_seed


TDLearningRes = namedtuple(
    "TDLearningRes", ["alpha", "gamma", "mean_reward", "mean_step", "test_mean_rewards", "test_episodes", "q_function", "policy_seed", "env_seed"])


def make_grid(x_axis_values, y_axis_values, x_values,
              y_values, z_values, fill_value: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            text = ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", color="black")

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
    player_sum = np.arange(player_axis_start, 21 + 1)
    dealer_show = np.arange(dealer_axis_start, 10 + 1)

    _, _, rewards_data = make_grid(player_sum, dealer_show, data["player_sum"].to_numpy(
    ), data["dealer_open_card"].to_numpy(), data["reward"].to_numpy())

    heatmap(rewards_data, row_labels=list(map(str, dealer_show)),
            col_labels=list(map(str, player_sum)), ax=ax, cmap="rainbow")

    ax.set_xlabel("?????????? ???????? ????????????")
    ax.set_ylabel("???????????????? ?????????? ??????????????")
    ax.set_title(title)

    return fig


def simulate(td_learning_cls, alpha: float, gamma: float, env, policy, is_learning: bool,
             action_space, total_episodes: int, total_test_episodes: int, num_run: int, total_runs: int) -> TDLearningRes:
    logger = logging.getLogger()
    q_function = policy.q_function
    td_learning = td_learning_cls(env, policy, q_function, alpha=alpha,
                                  gamma=gamma, is_learning=is_learning, action_space=action_space)
    td_res = td_learning.simulate(total_episodes, total_test_episodes)

    logger.info("Complete hyperparameter tuning %d of %d", num_run, total_runs)
    logger.info("Alpha: %f Gamma: %f. Mean reward: %f", alpha, gamma, td_res.mean_reward)

    return TDLearningRes(alpha, gamma, td_res.mean_reward,
                         td_res.mean_step, td_res.test_mean_rewards, td_res.test_episodes,
                         q_function, policy.seed, env.seed_value)


def generate_stat(td_learning_cls, gammas: Iterable[float], alpha: Iterable[float],
                  total_episodes: int, total_test_episodes: Optional[int], policy, is_learning: bool,
                  env, action_space, num_workers: int = 4) -> List[TDLearningRes]:
    values = []

    alpha_gamma = list(zip(*tuple(product(alpha, gammas))))
    alphas = alpha_gamma[0]
    gammas = alpha_gamma[1]
    del alpha_gamma
    envs = [copy.deepcopy(env) for _ in range(len(alphas))]
    policies = [copy.deepcopy(policy) for _ in range(len(alphas))]
    is_learnings = repeat(is_learning, len(alphas))
    action_spaces = repeat(action_space, len(alphas))
    total_episodes_values = repeat(total_episodes, len(alphas))
    total_test_episodes_values = repeat(total_test_episodes, len(alphas))
    td_learning_cls = repeat(td_learning_cls, len(alphas))
    num_runs = range(1, len(alphas) + 1)
    total_runs = repeat(len(alphas), len(alphas))

    with Pool(num_workers) as workers:
        values.extend(workers.starmap(simulate, zip(td_learning_cls, alphas, gammas, envs, policies,
                                                    is_learnings, action_spaces,
                                                    total_episodes_values,
                                                    total_test_episodes_values,
                                                    num_runs,
                                                    total_runs)))

    return values


def plot_td_learning_stat(alpha_axis: np.ndarray, gamma_axis: np.ndarray, results: List[TDLearningRes], title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    values = np.array(list(map(lambda x: (x.alpha, x.gamma, x.mean_reward), results)))

    alphas = values[:, 0]
    gammas = values[:, 1]
    mean_rewards = values[:, 2]

    _, _, mean_reward = make_grid(alpha_axis, gamma_axis, alphas, gammas, mean_rewards)

    def format(x):
        return f"{x:.4f}"
    heatmap(mean_reward, row_labels=list(
        map(format, gamma_axis)), col_labels=list(map(format, alpha_axis)), ax=ax)

    ax.set_xlabel("???????????????? ???????????????? alpha")
    ax.set_ylabel("gamma")
    ax.set_title(title)

    return fig


def plot_training_process(episodes: np.ndarray, rewards: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    _, trend = sm.tsa.filters.hpfilter(rewards)
    ax.plot(episodes, rewards)
    ax.plot(episodes, trend, label="??????????", c="red")
    ax.set_title("?????????????????? ?????????????? ?????????????? ???? ?????????? ????????????????")
    ax.set_xlabel("?????????? ??????????????")
    ax.set_ylabel("?????????????? ??????????????")
    ax.grid(True)
    plt.legend()

    return fig


def plot_stat(alpha_values: np.ndarray, gamma_values: np.ndarray, train_stat: List[TDLearningRes],
              test_stat: List[TDLearningRes]):
    train_fig = plot_td_learning_stat(alpha_values, gamma_values,
                                      train_stat, "?????????????? ?????????????? ???? ?????????? ????????????????")

    test_fig = plot_td_learning_stat(
        alpha_values, gamma_values, test_stat, "?????????????? ?????????????? ?????? ?????????????????????????? ?????????????????? ??????????????????")

    return train_fig, test_fig


def evaluate_td_learning(*, alpha_values: np.ndarray, gamma_values: np.ndarray, epsilon: float,
                         num_train_episodes: int, num_test_episodes: int, env, num_train_episodes_on_best_params: int, learning_cls=QLearningSimulation, action_space=Action,
                         ) -> Tuple[List[TDLearningRes], List[TDLearningRes], TDLearningRes, TDLearningRes]:

    logger = logging.getLogger()

    logger.info("Train on hyperparameters")

    q_function = QTableDict()

    policy = EpsilonGreedyPolicy(q_function, epsilon=epsilon, seed=get_seed())

    train_stat = generate_stat(learning_cls, gamma_values, alpha_values,
                               num_train_episodes, -1, policy, is_learning=True, env=env, action_space=action_space)

    best_stat = max(train_stat, key=lambda x: np.mean(x.test_mean_rewards))
    logger.info("Validate on the best hyperparameters alpha: %f gamma: %f",
                best_stat.alpha, best_stat.gamma)

    test_stat = generate_stat(learning_cls, gamma_values, alpha_values,
                              num_train_episodes, None, policy, is_learning=False, env=env, action_space=action_space)

    logger.info("Train on best parameters with steps: %d alpha: %f gamma: %f",
                num_train_episodes_on_best_params, best_stat.alpha, best_stat.gamma)
    train_q_function = QTableDict()

    policy = EpsilonGreedyPolicy(train_q_function, epsilon=epsilon, seed=best_stat.policy_seed)
    env.seed(env.seed_value)

    training_process = simulate(learning_cls, best_stat.alpha, best_stat.gamma,
                                env, policy, is_learning=True, action_space=action_space,
                                total_episodes=num_train_episodes_on_best_params, total_test_episodes=num_test_episodes, num_run=1, total_runs=1)

    policy = EpsilonGreedyPolicy(train_q_function, epsilon=0.0, seed=best_stat.policy_seed)
    # env.seed(env.seed_value)
    logger.info("Validate on best parameters with steps: %d", num_train_episodes_on_best_params)
    best_valid_stat = generate_stat(learning_cls, [best_stat.gamma], [best_stat.alpha],
                                    num_train_episodes_on_best_params, None, policy, is_learning=False, env=env, action_space=action_space)[0]

    return train_stat, test_stat, training_process, best_valid_stat


def plot_final_stat(states: dict):
    translate = {"base": "?????????????? ??????????????????", "double": "?????????????????? ?? ??????????????????",
                 "counting": "?????????????????? ?? ?????????????????? ????????", "q": "Q-????????????????", "sarsa": "Sarsa"}

    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0
    objects = []
    labels = []

    for i, env in enumerate(states):
        for algo, color, shift in zip(("q", "sarsa"), ("blue", "green"), (-0.2, 0.2)):
            bar = plt.bar(i + shift, -states[env][algo].mean_reward, width=0.4, color=color)
            if i == 0:
                objects.append(bar)
                labels.append(f"????????????????: {translate[algo]}")
        i += 1
    ax.set_xticks(tuple(range(len(states))))
    ax.set_xticklabels(list(map(lambda x: translate[x], states)))
    ax.legend(objects, labels)
    ax.set_title("?????????????? ?????????? ???????????????????? ???? -1. ?????????? ?? 0 ??????????.")

    return fig

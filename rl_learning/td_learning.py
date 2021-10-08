from typing import Tuple, Optional
import random
import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod

import tqdm
from gym.envs.toy_text import BlackjackEnv

from .utils import Action, QTableDict

TDLearningRes = namedtuple("TDLearningRes", ["mean_reward", "mean_step", "test_mean_rewards"])


class TDLearning(ABC):
    def __init__(self, env: BlackjackEnv, policy, q_table: QTableDict, **kwargs):
        self._env = env
        self._policy = policy
        self._init_qtable(q_table)
        self._q_table = q_table
        self._is_learning = kwargs["is_learning"]

    def _init_qtable(self, q_table: QTableDict):
        for action in (Action.hit, Action.stick):
            for ace in (0, 1):
                for dealer_open_card in range(1, 11):
                    for player_sum in range(2, 32):
                        q_table.set_value(
                            (player_sum, dealer_open_card, ace), action, random.uniform(-1, 1))

    @abstractmethod
    def _update_q_function(self, old_state, action: int, new_state, reward: float, new_action: int):
        pass

    def _generate_episode(self) -> Tuple[float, int]:
        state = self._env.reset()

        total_rewards = 0
        total_steps = 0

        done = False

        while not done:
            action = self._policy.action(state)
            old_state = state
            state, reward, done, _ = self._env.step(action)

            total_rewards += reward
            total_steps += 1

            self._update_q_function(old_state, action, state, reward, None)

        return total_rewards, total_steps

    def simulate(self, num_episodes: int, num_policy_exp: Optional[int] = None) -> TDLearningRes:
        all_reawrds = []
        step_counts = []
        test_rewards = []

        for _ in tqdm.trange(num_episodes):
            self._is_learning = True
            reward, steps = self._generate_episode()
            all_reawrds.append(reward)
            step_counts.append(steps)

            if num_policy_exp is not None:
                self._is_learning = False
                exp_rewards = []

                for _ in range(num_policy_exp):
                    reward, steps = self._generate_episode()
                    exp_rewards.append(reward)

                test_rewards.append(np.mean(exp_rewards))

        return TDLearningRes(np.mean(all_reawrds), np.mean(step_counts), np.array(test_rewards))


class QLearningSimulation(TDLearning):
    def __init__(self, env: BlackjackEnv, policy, q_table: QTableDict, **kwargs):
        super().__init__(env, policy, q_table, **kwargs)
        self._alpha = kwargs["alpha"]
        self._gamma = kwargs["gamma"]

    def _update_q_function(self, old_state, action: int, new_state, reward: float, new_action: int):
        if self._is_learning:
            old_value = self._q_table.get_value(old_state, action)
            greedy_reward = self._gamma * max(self._q_table.get_actions(new_state).values())
            self._q_table.set_value(old_state, action, old_value + self._alpha *
                                    (reward + greedy_reward - old_value))


class Sarsa(TDLearning):
    def __init__(self, env: BlackjackEnv, policy, q_table: QTableDict, **kwargs):
        super().__init__(env, policy, q_table, **kwargs)
        self._alpha = kwargs["alpha"]
        self._gamma = kwargs["gamma"]

    def _update_q_function(self, old_state, action: int, new_state, reward: float, new_action: int):
        if self._is_learning:
            old_value = self._q_table.get_value(old_state, action)
            greedy_reward = self._gamma * self._q_table.get_value(new_state, new_action)
            self._q_table.set_value(old_state, action, old_value + self._alpha *
                                    (reward + greedy_reward - old_value))

    def _generate_episode(self) -> Tuple[float, int]:
        state = self._env.reset()

        total_rewards = 0
        total_steps = 0

        done = False

        action = self._policy.action(state)

        while not done:
            old_state = state
            state, reward, done, _ = self._env.step(action)

            old_action = action
            action = self._policy.action(state)

            total_rewards += reward
            total_steps += 1

            self._update_q_function(old_state, old_action, state, reward, action)

        return total_rewards, total_steps

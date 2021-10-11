from typing import Tuple, Optional
import random
import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
import logging
from itertools import product


import log_set
from .utils import Action, QTableDict
from .black_jask_modified import BaseBlackjackEnv

TDLearningRes = namedtuple("TDLearningRes", ["mean_reward", "mean_step", "test_mean_rewards"])


class TDLearning(ABC):
    def __init__(self, env: BaseBlackjackEnv, policy, q_table: QTableDict, action_space=Action, **kwargs):
        self._env = env
        self._policy = policy
        self._action_space = action_space
        self._q_table = q_table
        self._is_learning = kwargs["is_learning"]
        self._logger = logging.getLogger("simulation")

    def _init_qtable(self, q_table: QTableDict):
        for action in tuple(self._action_space):
            space = self._env.observation_space_values()
            for state in product(*space.values()):
                q_table.set_value(state, action, random.uniform(-1, 1))

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
        self._init_qtable(self._q_table)

        all_rewards = []
        step_counts = []
        test_rewards = []

        print_every = num_episodes // 10

        for episode in range(num_episodes):
            self._is_learning = True
            reward, steps = self._generate_episode()
            all_rewards.append(reward)
            step_counts.append(steps)

            if num_policy_exp is not None:
                self._is_learning = False
                exp_rewards = []

                for _ in range(num_policy_exp):
                    reward, steps = self._generate_episode()
                    exp_rewards.append(reward)

                test_rewards.append(np.mean(exp_rewards))

            if (episode + 1) % print_every == 0:
                self._logger.info(f"Progress: {episode / num_episodes:.2%}")

        return TDLearningRes(np.mean(all_rewards), np.mean(step_counts), np.array(test_rewards))


class QLearningSimulation(TDLearning):
    def __init__(self, env: BaseBlackjackEnv, policy, q_table: QTableDict,  action_space=Action, **kwargs):
        super().__init__(env, policy, q_table, action_space=action_space, **kwargs)
        self._alpha = kwargs["alpha"]
        self._gamma = kwargs["gamma"]

    def _update_q_function(self, old_state, action: int, new_state, reward: float, new_action: int):
        if self._is_learning:
            old_value = self._q_table.get_value(old_state, action)
            greedy_reward = self._gamma * max(self._q_table.get_actions(new_state).values())
            self._q_table.set_value(old_state, action, old_value + self._alpha *
                                    (reward + greedy_reward - old_value))


class Sarsa(TDLearning):
    def __init__(self, env: BaseBlackjackEnv, policy, q_table: QTableDict, action_space=Action, **kwargs):
        super().__init__(env, policy, q_table, action_space=action_space, **kwargs)
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

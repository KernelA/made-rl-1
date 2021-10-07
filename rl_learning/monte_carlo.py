from typing import List, Tuple, Dict
from collections import defaultdict

import tqdm
from gym.envs.toy_text import BlackjackEnv


class MonetCarloSimulation:
    def __init__(self, env: BlackjackEnv, policy):
        self._env = env
        self._strategy = policy

    def _generate_episode(self) -> Tuple[List, List, List]:
        states, actions, rewards = [], [], []
        state = self._env.reset()

        done = False

        while not done:
            states.append(state)
            action = self._strategy.action(state)
            actions.append(action)
            state, reward, done, _ = self._env.step(action)
            rewards.append(reward)

        return states, actions, rewards

    def simulate(self, num_episodes: int) -> Dict[tuple, float]:
        value_function = defaultdict(float)
        state_counter = defaultdict(int)

        for _ in tqdm.trange(num_episodes):
            states, _, rewards = self._generate_episode()

            total_reward_at_time = 0

            for time in range(len(states) - 1, -1, -1):
                reward = rewards[time]
                state = states[time]
                total_reward_at_time += reward

                if state not in states[:time]:
                    state_counter[state] += 1
                    value_function[state] += (total_reward_at_time -
                                              value_function[state]) / state_counter[state]

        return value_function

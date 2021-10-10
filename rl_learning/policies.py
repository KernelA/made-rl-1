import random

from .utils import Action, QTableDict


class SimplePolicy:
    def action(self, state) -> Action:
        if state[0] >= 19:
            return Action.stick
        else:
            return Action.hit


class EpsilonGreedyPolicy:
    def __init__(self, q_function: QTableDict, epsilon: float):
        self._q_function = q_function
        self._epsiolon = epsilon

    def action(self, state) -> Action:
        if random.random() < self._epsiolon:
            return random.choice((Action.stick, Action.hit))
        else:
            return max(self._q_function.get_actions(state).items(), key=lambda x: x[1])[0]

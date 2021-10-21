import random


from .utils import Action, QTableDict


class SimplePolicy:
    def action(self, state) -> Action:
        if state[0] >= 19:
            return Action.stick
        else:
            return Action.hit


class EpsilonGreedyPolicy:
    def __init__(self, q_function: QTableDict, epsilon: float, seed: int):
        self.q_function = q_function
        self._epsiolon = epsilon
        self._generator = random.Random(seed)
        self.seed = seed

    def action(self, state) -> Action:
        if self._generator.random() < self._epsiolon:
            return random.choice((Action.stick, Action.hit))
        else:
            return max(self.q_function.get_actions(state).items(), key=lambda x: x[1])[0]

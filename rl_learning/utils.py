import enum
from collections import UserDict
from typing import Any, Dict, Tuple


@enum.unique
class Action(enum.IntEnum):
    stick = 0
    hit = 1


@enum.unique
class ExtendedAction(enum.IntEnum):
    stick = 0
    hit = 1
    double = 2


class QTableDict(UserDict):
    """Represnet discrete Q(s, a) tabular function
    """

    def __init__(self):
        super().__init__(dict())

    def set_value(self, state: Tuple[Any], action, value: float):
        if state not in self.data:
            self.data[state] = {action: value}
        else:
            self.data[state][action] = value

    def get_actions(self, state) -> Dict[int, float]:
        return self.data[state]

    def get_value(self, state, action) -> float:
        return self.data[state][action]

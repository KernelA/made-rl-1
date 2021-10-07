import enum
from collections import UserDict


class Action(enum.IntEnum):
    hit = 1
    stick = 0


class QTableDict(UserDict):
    """Represnet discrete Q(s, a) tabular function
    """

    def __init__(self):
        super().__init__(dict())

    def set_value(self, state, action, value):
        if state not in self.data:
            self.data[state] = {action: value}
        else:
            self.data[state][action] = value

    def get_actions(self, state):
        return self.data[state]

    def get_value(self, state, action):
        return self.data[state][action]

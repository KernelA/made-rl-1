from simultaion import Action


class SimplePolicy:
    def action(self, state) -> Action:
        if state[0] >= 19:
            return Action.stick
        else:
            return Action.hit

    def init_action(self) -> Action:
        return Action.hit

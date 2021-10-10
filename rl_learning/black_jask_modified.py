from gym.envs.toy_text import BlackjackEnv
from gym.envs.toy_text.blackjack import cmp, is_bust, draw_card, score, is_natural, sum_hand
from gym import spaces

from .utils import ExtendedAction


class BlackjackEnvDouble(BlackjackEnv):
    def __init__(self, natural=False):
        super().__init__(natural=natural)
        self.action_space = spaces.Discrete(3)
        self._reward_multiplier = 1

    def reset(self):
        super().reset()
        self._reward_multiplier = 1
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)

        if action == ExtendedAction.double:
            self._reward_multiplier *= 2
            action = ExtendedAction.hit

        if action == ExtendedAction.hit:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        return self._get_obs(), self._reward_multiplier * reward, done, {}

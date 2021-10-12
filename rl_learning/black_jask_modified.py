import random
from typing import Dict, List

import gym
from gym.envs.toy_text.blackjack import cmp, is_bust, draw_card, score, is_natural, sum_hand, deck, usable_ace
from gym import spaces
from gym.spaces.discrete import Discrete
from gym.utils import seeding

from .utils import ExtendedAction


class BaseBlackjackEnv(gym.Env):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _draw_card(self, np_random):
        return draw_card(np_random)

    def observation_space_values(self) -> dict:
        return {"player_sum": tuple(range(2, 32)), "dealer_open_card": tuple(range(1, 11)), "usable_ace": (True, False)}

    def _draw_hand(self, np_random):
        return [self._draw_card(np_random), self._draw_card(np_random)]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(self._draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self._draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = self._draw_hand(self.np_random)
        self.player = self._draw_hand(self.np_random)
        return self._get_obs()


class BlackjackEnvDouble(BaseBlackjackEnv):
    def __init__(self, natural=False):
        super().__init__(natural=natural)
        self.action_space = spaces.Discrete(3)
        self._reward_multiplier = 1

    def reset(self):
        state = super().reset()
        self._reward_multiplier = 1
        return state

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


class BlackjackWithShuffle(BlackjackEnvDouble):
    """BlackJack with double reward and shuffle
    """
    CARD_WEIGHTS = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}

    def __init__(self, min_decks_before_shuffle: int, natural=False):
        self._min_decks_before_shuffle = min_decks_before_shuffle
        # Shift to map -4, -3, ... -> 0, 1, ...
        self._counting_shift = 4
        self._is_shuffle = False
        super().__init__(natural=natural)

        self._real_deck = self._get_deck()
        self._card_counting = 0

        new_obs_space = []

        for index in range(len(self.observation_space)):
            new_obs_space.append(self.observation_space[index])

        self._counting_space_size = 14

        # Computation space -4, -3, ..., 8
        new_obs_space.append(Discrete(self._counting_space_size))

        self.observation_space = spaces.Tuple(new_obs_space)

    def observation_space_values(self) -> dict:
        old_space = super().observation_space_values()
        old_space["card_counter"] = tuple(range(self._counting_space_size + 1))
        return old_space

    def _get_deck(self) -> List[int]:
        real_deck = deck * 4
        random.shuffle(real_deck, self._random)

        return real_deck

    def _draw_card(self, np_random):
        card = self._real_deck.pop()
        self._card_counting += self.CARD_WEIGHTS[card]
        if len(self._real_deck) < self._min_decks_before_shuffle and not self._is_shuffle:
            random.shuffle(self._real_deck, self._random)
            self._is_shuffle = True
        return card

    def _get_obs(self):
        base_state = super()._get_obs()
        return base_state + (self._card_counting + self._counting_shift, )

    def _random(self):
        return self.np_random.uniform()

    def reset(self):
        self._card_counting = 0
        self._is_shuffle = False
        self._real_deck = self._get_deck()
        res = super().reset()
        return res

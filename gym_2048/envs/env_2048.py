from game_2048.logic import Action, Game, game_step
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Env2048(gym.Env):

    def __init__(self, shape=(4, 4)):
        # parametrize the gym interface
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=524289, shape=shape, dtype=np.int32)
        self.metadata = {'render.modes': ['human', 'ansi']}
        self.reward_range = (-10., float('inf'))
        # init the game
        self.game = Game(shape)

    def __init__(self):

    def step(self, action):
        self.game = game_step(game, Action(action))

    def reset(self):

    def render(self, mode='human'):

    def close(self):

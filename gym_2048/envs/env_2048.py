import game_2048.logic as logic
import numpy as np
import gym
from gym import spaces
import sys
from six import StringIO


class Env2048(gym.Env):

    def __init__(self, shape=(4, 4)):
        # parametrize the gym interface
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2**32, shape=shape, dtype=np.uint32)
        self.metadata = {'render.modes': ['human', 'ansi']}
        self.reward_range = (-10., float('inf'))
        # init the game
        self.game = logic.Game(shape)

    def step(self, action) -> (object, float, bool, dict):
        '''Execute one action in the game.

        Parameters
        ----------
        action: object
            An action provided by the agent.

        Returns
        -------
        observation: object
            The agent's observation of the current environment.
        reward: float
            Amount of reward returned after previous action.
        done: bool
            Whether the episode has ended.
        info: dict
            contains auxiliary diagnostic information
        '''
        self.game, score, _ = logic.game_step(self.game, logic.Action(action))
        return self.game.board, score, self.game.finished, None

    def reset(self) -> object:
        """Resets the state of the environment and returns an initial observation.

        Returns
        -------
        observation: object
            The initial observation.
        """
        self.game.reset()
        return self.game.board

    def render(self, mode='human'):
        """Renders the environment.

        Parameters
        ----------
        mode: str
            - human: renders the board to the system output.
            - ansi: returns the string representation of the board."""
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(self.game.board)
        if mode != 'human':
            return outfile

    def seed(self, seed=None):
        """Sets the seed for this environments random number generator.

        Returns
        -------
        seed
            The main seed.
        """
        self.game.seed(seed)
        return seed
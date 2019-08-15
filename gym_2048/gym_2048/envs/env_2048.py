from gym_2048 import Action, Game, game_step, get_info
import numpy as np
import gym
from gym import spaces
import sys
from six import StringIO


class Env2048(gym.Env):
    '''The basic 2048 environment which outputs the raw board as observation.
    The score of each round is returned as reward so higher tile merges return
    higher rewards. The game is finished when no valid move is possible.'''

    def __init__(self, shape: (int, int) = (4, 4)):
        '''Creates a new game.

        Parameters
        ----------
        shape: (int, int)
            The shape of the board, must be two dimensional.'''
        # parametrize the gym interface
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2048, shape=shape, dtype=np.uint32)
        self.metadata = {'render.modes': ['human', 'ansi']}
        self.reward_range = (0, 2048)
        # init the game
        self.game = Game(shape)

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
            'score': int - the score of the game
            'high_tile': int - the highest tile in the game
            'steps: int - number of valid steps executed'''
        self.game, score, valid = game_step(
            self.game, Action(action))
        return self.game.board, score, self.game.finished, get_info(self.game)

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
        outfile.write(str(self.game.board)+'\n\n')
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

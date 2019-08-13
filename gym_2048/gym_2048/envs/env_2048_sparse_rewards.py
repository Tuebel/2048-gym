from gym_2048.envs.env_2048 import Env2048
from gym.spaces import Box


class Env2048SparseRewards(Env2048):
    '''A 2048 environment which outputs the raw board as observation. Only the
    final score is returned all other scores are zero. This challenges the
    algorithm but might lead to long term oriented policies. The game is
    finished when no valid move is possible.'''

    def __init__(self, shape: (int, int) = (4, 4)):
        '''Creates a new game.

        Parameters
        ----------
        shape: tuple
            The shape of the board, must be two dimensional.'''
        super(Env2048SparseRewards, self).__init__(shape)
        # parametrize the gym interface
        self.observation_space = Box(
            low=0, high=2**16, shape=shape, dtype=np.uint32)

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
        self.game, score, valid = game_2048.game_step(
            self.game, game_2048.Action(action))
        if self.game.finished:
            return self.game.board, self.game.score, self.game.finished, None
        else:
            return self.game.board, 0, self.game.finished, None
from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np


class OneChannel(ObservationWrapper):
    '''Wraps observations by adding a channel dimension.
    For example a (4, 4) shaped observation is reshaped into a (4, 4, 1)
    shape.'''

    def __init__(self, env):
        super(OneChannel, self).__init__(env)
        assert isinstance(self.observation_space, Box), (
            "OneChannel wrapper is only usable with box observations.")
        self.observation_space.shape = (*self.observation_space.shape, 1)

    def observation(self, observation):
        return np.reshape(observation, self.observation_space.shape)

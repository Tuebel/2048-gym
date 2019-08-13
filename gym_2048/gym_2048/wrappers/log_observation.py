from gym import ObservationWrapper
from gym.spaces import Box
import numpy as np


class LogObservation(ObservationWrapper):
    '''Applies a logarithm base 2 on the observation to scale it.
    Intermediately replaces 0 with 1 so the log(1)=0 again.'''

    def __init__(self, env):
        super(LogObservation, self).__init__(env)
        assert isinstance(self.observation_space, Box), (
            "OneChannel wrapper is only usable with box observations.")

    def observation(self, observation):
        obs = np.copy(observation)
        # avoid log(0)
        obs[obs == 0] = 1
        return np.log2(obs)

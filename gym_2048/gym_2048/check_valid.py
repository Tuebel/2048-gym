from gym_2048 import Action, merge
import numpy as np


def check_valid(state: np.array, action: Action):
    state = np.reshape(state, state.shape[0:2])
    _, _, valid = merge(state, action)
    return valid

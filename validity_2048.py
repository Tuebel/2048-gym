from game_2048 import Action, merge
import numpy as np


def check_valid_2048(state: np.array, action: Action):
    _, _, valid = merge(state, action)
    return valid

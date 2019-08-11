from huskarl.policy import Policy
from game_2048 import Action, merge
import random
import numpy as np


def check_valid_2048(state: np.array, action: Action):
    _, _, valid = merge(state, action)
    return valid


class EpsGreedy2048(Policy):
    """This policy acts like any other eps-greedy policy except it"""

    def __init__(self, eps, check_valid=check_valid_2048):
        self.eps = eps
        self.check_valid = check_valid

    def act(self, qvals, state):
        # return the valid action with the highest Q value
        if random.random() > self.eps:
            for i in range(len(qvals)):
                action = np.argmax(qvals)
                if self.check_valid(state, action):
                    return action
                else:
                    qvals[action] = 0
        # return any valid action
        while True:
            action = random.randrange(len(qvals))
            if self.check_valid(state, action):
                return action

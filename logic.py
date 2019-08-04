from enum import Enum, auto
import numpy as np


class State():
    def __init__(self):
        self.board = np.zeros([4, 4])


class Action(Enum):
    '''The state of the game is represented as two dimensional numpy array.
    It is row major.'''
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


def first_free(row):
    '''Searches the index of the first free element (0) in a row

    Paramters
    ---------
    row: numpy.array
      The row to analyze

    Returns
    -------
    index: int
      The index of the first free element. Size of row if not free element
      found'''
    for i in range(0, row.size):
        if row[i] == 0:
            return i
    return row.size


def merge_row(row):
    '''Merges one row from left to right. Transform the board apply the merge
    and transform it back afterwards.

    Paramters
    ---------
    row: numpy.array
      The row to merge

    Returns
    -------
    row: numpy.array
      The merged row'''
    last_merge = -1
    for i in range(0, row.size):
        # move as far left as possible
        new_index = first_free(row)
        if new_index < i:
            row[new_index] = row[i]
            row[i] = 0
        else:
            new_index = i
        # merge possible?
        if (last_merge < new_index - 1
                and row[new_index - 1] == row[new_index]):
            row[new_index - 1] = 2 * row[new_index - 1]
            row[new_index] = 0
    return row


def merge(state, action):
    '''Merges the rows or columns of the board depending on the action taken.

    Parameters
    ----------
    state: State
      The state of the current board
    action: Action
      Direction to merge
    '''
    # iteration will always be: rows, colums from
    # if action is Action.LEFT


a = np.array([8, 4, 2, 0, 2, 2])
merge_row(a)
print(a)

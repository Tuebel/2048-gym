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
    '''Searches the index of the first free element (0) in a row.

    Paramters
    ---------
    row: numpy.array
        The row to analyze.

    Returns
    -------
    index: int
        The index of the first free element. Size of row if not free element
        found.'''
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
        The row to merge.

    Returns
    -------
    row: numpy.array
        The merged row.
    n_merges: int
        Number of successfull merges.'''
    last_merge = -1
    n_merges = 0
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
            n_merges += 1
    return row, n_merges


def transform_before_merge(board, action):
    '''Transforms the board so the the board can be merged row by row.

    Parameters
    ----------
    board: numpy.array
        The board to transform.
    action: Action
        The action that will be executed.

    Returns
    -------
    board: numpy.array
        The transformed board.'''
    # LEFT does not require transformation
    if action == Action.RIGHT:
        board = np.fliplr(board)
    elif action == Action.UP:
        board = np.transpose(board)
    elif action == Action.DOWN:
        board = np.transpose(board)
        board = np.fliplr(board)
    return board


def transform_after_merge(board, action):
    '''Transforms the board after the merge has been executed

    Parameters
    ----------
    board: numpy.array
        The board to transform.
    action: Action
        The action that has been executed

    Returns
    -------
    board: numpy.array
        The transformed board.'''
    # LEFT does not require transformation
    if action == Action.RIGHT:
        board = np.fliplr(board)
    elif action == Action.UP:
        board = np.transpose(board)
    elif action == Action.DOWN:
        board = np.fliplr(board)
        board = np.transpose(board)
    return board


def merge(board, action):
    '''Merges the rows or columns of the board depending on the action taken.

    Parameters
    ----------
    board: numpy.array
        The state of the current board.
    action: Action
        Direction to merge.

    Returns
    -------
    board: numpy.array
        The board after the merge action has been executed.
    n_merges: int
        Number of successful merges.
    '''
    n_merges = 0
    board = transform_before_merge(board, action)
    n_rows, _ = board.shape
    for row in range(0, n_rows):
        board[row, :], n_m = merge_row(board[row, :])
        n_merges += n_m
    board = transform_after_merge(board, action)
    return board, n_merges

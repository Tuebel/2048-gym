from copy import deepcopy
from enum import Enum, auto
import numpy as np
from random import randrange, random
'''The logic of the game 2048.
Purely functional so copies are created of the state.'''


class Game:
    '''The state of a game is represented by the board (numpy array).
    Additionally the score must be stored and the information if the game is
    finished'''

    def __init__(self, shape=(4, 4)):
        '''Generates a new game with a board and score

        Parameters
        ----------
        shape: tuple
            The dimensions of the board.
        '''
        self.board = new_board(shape)
        self.score = 0
        self.finished = False

    def __repr__(self):
        return f'{self.board}\n{self.score}\n{self.finished}'


class Action(Enum):
    '''The state of the game is represented as two dimensional numpy array.
    It is row major.'''
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


def game_step(game: Game, action: Action) -> Game:
    '''Executes one game step for the given game.

    Parameters
    ----------
    game: Game
        The current state of the game.
    action:
        The action to execute now.

    Returns
    -------
    game: Game
        The new state of the game.'''
    game = deepcopy(game)
    game.board, score, valid = merge(game.board, action)
    # if action has not been valid the game state didn't change
    if valid:
        game.score += score
        game.board = generate_element(game.board)
        game.finished = is_finished(game.board)
    return game


def generate_element(board: np.array) -> np.array:
    '''Randomly generates a 2 or 4 on the board on an empty field.

    Parameters
    ----------
    board: numpy.array
        The board with empty fields.

    Returns
    -------
    board: numpy.array
        The board with a new randomly generated number.'''
    board = np.copy(board)
    zero_elements = np.where(board == 0)
    random_index = randrange(0, len(zero_elements[0]))
    random_position = (
        zero_elements[0][random_index], zero_elements[1][random_index])
    if random() < 0.9:
        board[random_position] = 2
    else:
        board[random_position] = 4
    return board


def new_board(shape: tuple) -> np.array:
    '''Generates a new board with two randomly generated elements.

    Parameters
    ----------
    shape: tuple
        The shape of the board.

    Returns
    -------
    board: numpy.array
        The new board with two random elements.'''
    board = np.zeros(shape)
    board = generate_element(board)
    board = generate_element(board)
    return board


def first_free_in_row(row: np.array) -> int:
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


def transform_before_merge(board: np.array, action: Action) -> np.array:
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


def transform_after_merge(board: np.array, action: Action) -> np.array:
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


def merge_row(row: np.array) -> (np.array, int, bool):
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
    score: int
        Sum of the merged numbers.
    valid: bool
        True if any element has been moved. False otherwise.'''
    # The whole API is purely functional
    row = np.copy(row)
    last_merge = -1
    score = 0
    valid = False
    for i in range(0, row.size):
        if row[i] == 0:
            continue
        # move as far left as possible
        new_index = first_free_in_row(row)
        if new_index < i:
            row[new_index] = row[i]
            row[i] = 0
            valid = True
        else:
            new_index = i
        # merge possible?
        if (last_merge < new_index - 1
                and row[new_index - 1] == row[new_index]):
            row[new_index - 1] = 2 * row[new_index - 1]
            row[new_index] = 0
            score += row[new_index - 1]
            valid = True
    return row, score, valid


def merge(board: np.array, action: Action) -> (np.array, int, bool):
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
    score: int
        Sum of the merged numbers within this action.
    valid: bool
        True if any element has been moved. False otherwise.'''
    # purely function API
    board = np.copy(board)
    score = 0
    valid = False
    board = transform_before_merge(board, action)
    n_rows, _ = board.shape
    for row in range(0, n_rows):
        board[row, :], row_score, row_valid = merge_row(board[row, :])
        score += row_score
        valid |= row_valid
    board = transform_after_merge(board, action)
    return board, score, valid


def is_finished(board: np.array) -> bool:
    '''Checks if any action is possible: Any zero or score is possible

    Parameters
    ----------
    board: numpy.array
        The board to check.

    Returns
    -------
    result: bool
        True if no action is possible, false otherwise.'''
    if len(np.where(board == 0)[0]) > 0:
        return False
    for action in Action:
        _, _, valid = merge(board, action)
        if valid:
            return False
    return True

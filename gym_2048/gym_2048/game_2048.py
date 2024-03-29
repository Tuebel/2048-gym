from copy import deepcopy
from enum import Enum
from random import Random
import numpy as np
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
        self.random = Random()
        # state of the game
        self.board = new_board(shape, self.random)
        # score = sum of merged number
        self.score = 0
        # number of valid steps executed
        self.steps = 0
        # no valid move possible
        self.finished = False

    def __repr__(self):
        return f'{self.board}\nScore {self.score}\nFinished {self.finished}\n'

    def reset(self):
        '''Resets the game to the initial state.'''
        self.board = new_board(self.board.shape, self.random)
        self.score = 0
        self.steps = 0
        self.finished = False

    def seed(self, seed=None):
        '''Set the seed of the random generator.'''
        self.random.seed(seed)


class Action(Enum):
    '''The possible moves in the game as Enum.'''
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


def game_step(game: Game, action: Action) -> (Game, int, bool):
    '''Executes one game step for the given game.

    Parameters
    ----------
    game: Game
        The current state of the game.
    action: Action
        The action to execute now.

    Returns
    -------
    game: Game
        The new state of the game.
    score: int
        The score of this round.
    valid: bool
        If the action resulted in a valid move.'''
    game = deepcopy(game)
    game.board, score, valid = merge(game.board, action)
    # if action has not been valid the game state didn't change
    if valid:
        game.score += score
        game.steps += 1
        game.board = generate_element(game.board, game.random)
        game.finished = is_finished(game.board)
    return game, score, valid


def generate_element(board: np.array, random: Random) -> np.array:
    '''Randomly generates a 2 or 4 on the board on an empty field.

    Parameters
    ----------
    board: numpy.array
        The board with empty fields.
    random: Random
        The random number generator

    Returns
    -------
    board: numpy.array
        The board with a new randomly generated number.'''
    board = np.copy(board)
    zero_elements = np.where(board == 0)
    random_index = random.randrange(0, len(zero_elements[0]))
    random_position = (
        zero_elements[0][random_index], zero_elements[1][random_index])
    if random.random() < 0.9:
        board[random_position] = 2
    else:
        board[random_position] = 4
    return board


def new_board(shape: (int, int), random: Random) -> np.array:
    '''Generates a new board with two randomly generated elements.

    Parameters
    ----------
    shape: (int, int)
        The shape of the board.
    random: Random
        The random number generator

    Returns
    -------
    board: numpy.array
        The new board with two random elements.'''
    board = np.zeros(shape)
    board = generate_element(board, random)
    board = generate_element(board, random)
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
        The index of the first free element. -1 if not free element found.'''
    for i in range(0, row.size):
        if row[i] == 0:
            return i
    return -1


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
    action = Action(action)
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
    action = Action(action)
    if action == Action.RIGHT:
        board = np.fliplr(board)
    elif action == Action.UP:
        board = np.transpose(board)
    elif action == Action.DOWN:
        board = np.fliplr(board)
        board = np.transpose(board)
    return board


def fill_gaps_in_row(row: np.array) -> (np.array, bool):
    '''Fills the zero entries by moving everything to the left.

    Parameters
    ----------
    row: np.array
        The row to fill up.

    Returns
    -------
    row: np.array
        The merged row.
    valid: bool
        If any nonzero movement has been executed.'''
    row = np.copy(row)
    valid = False
    for i in range(0, row.size):
        if row[i] > 0:
            first_free = first_free_in_row(row)
            if first_free >= 0 and first_free < i:
                row[first_free] = row[i]
                row[i] = 0
                valid = True
    return row, valid


def merge_row(row: np.array) -> (np.array, int, bool):
    '''Merges one row from right to left. Transform the board apply the merge
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
    score = 0
    row, valid = fill_gaps_in_row(row)
    # merge neighbors
    for i in range(1, row.size):
        if row[i-1] == row[i] and row[i] > 0:
            row[i-1] += row[i]
            row[i] = 0
            score += row[i-1]
            valid = True
    row, _ = fill_gaps_in_row(row)
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
        merged_row, row_score, row_valid = merge_row(board[row, :])
        board[row] = merged_row
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


def highest_tile(board: np.array) -> int:
    '''Finds the highest tile number

    Parameters
    ----------
    board: numpy.array
        Find the highest tile of this board.

    Returns
    -------
    result: int
        The highest tile number in the game.'''
    return np.amax(board)


def get_info(game: Game) -> dict:
    '''Returns infos about the current game state

    Parameters
    ----------
    game: Game
        The game to describe.

    Returns
    -------
    dict: {'score': int, 'high_tile': int}
        The score and highest tile number of the game.'''
    h_tile = highest_tile(game.board)
    return {'score': game.score, 'high_tile': h_tile, 'steps': game.steps}

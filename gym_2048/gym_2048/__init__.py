from .game_2048 import Action
from .game_2048 import Game
from .game_2048 import game_step
from .game_2048 import is_finished
from .game_2048 import merge
# registration of the gym envs
from gym.envs.registration import register

register(
    id='2048-2x2-v0',
    entry_point='gym_2048.envs:Env2048',
    kwargs={'shape': (2, 2)})
register(
    id='2048-2x2-v1',
    entry_point='gym_2048.envs:Env2048LogTwo',
    kwargs={'shape': (2, 2)})
register(
    id='2048-2x2-v2',
    entry_point='gym_2048.envs:Env2048SparseRewards',
    kwargs={'shape': (2, 2)})
register(
    id='2048-3x3-v0',
    entry_point='gym_2048.envs:Env2048',
    kwargs={'shape': (3, 3)})
register(
    id='2048-3x3-v1',
    entry_point='gym_2048.envs:Env2048LogTwo',
    kwargs={'shape': (3, 3)})
register(
    id='2048-3x3-v2',
    entry_point='gym_2048.envs:Env2048SparseRewards',
    kwargs={'shape': (3, 3)})
register(
    id='2048-4x4-v0',
    entry_point='gym_2048.envs:Env2048',
    kwargs={'shape': (4, 4)})
register(
    id='2048-4x4-v1',
    entry_point='gym_2048.envs:Env2048LogTwo',
    kwargs={'shape': (4, 4)})
register(
    id='2048-4x4-v2',
    entry_point='gym_2048.envs:Env2048SparseRewards',
    kwargs={'shape': (4, 4)})

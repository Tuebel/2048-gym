from gym.envs.registration import register

register(
    id='2048-4x4-v0',
    entry_point='gym_2048.envs:Env2048',
    kwargs={'shape': (4, 4), 'max_invalid_moves': 500})
register(
    id='2048-4x4-v1',
    entry_point='gym_2048.envs:Env2048LogTwo',
    kwargs={'shape': (4, 4), 'max_invalid_moves': 500})
register(
    id='2048-4x4-v2',
    entry_point='gym_2048.envs:Env2048SparseRewards',
    kwargs={'shape': (4, 4), 'max_invalid_moves': 500})

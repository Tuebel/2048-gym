from gym.envs.registration import register

register(
    id='2048-3x3-v0',
    entry_point='gym_2048.envs:Env2048',
    kwargs={'shape': (3, 3)})
register(
    id='2048-4x4-v0',
    entry_point='gym_2048.envs:Env2048',
    kwargs={'shape': (4, 4)})

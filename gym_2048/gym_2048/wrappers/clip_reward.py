from gym import RewardWrapper


class ClipReward(RewardWrapper):
    """Clips the rewards to 1:
    R = R for R < 1
    R = 1 for R >= 1"""

    def __init__(self, env):
        super(ClipReward, self).__init__(env)

    def reward(self, reward):
        if reward < 1:
            return reward
        else:
            return 1

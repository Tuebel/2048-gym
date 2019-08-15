import os


class Logger:
    '''Logs the info and reward of the 2048 simulations'''

    def __init__(self, n_instances):
        # for every instance store the episode results
        self.episode_high_tiles = [[] for i in range(n_instances)]
        self.episode_rewards = [[] for i in range(n_instances)]
        self.episode_scores = [[] for i in range(n_instances)]
        self.episode_steps = [[] for i in range(n_instances)]

    def log_episode(self, instance: int, observation: object, reward: float,
                    info: dict):
        '''Logs the information of a finished 2048 episode to the memory.

        Parameters
        ----------
        instance: int
            Id of the instance that has finished an episode.
        observation: object
            Final observation of the episode.
        reward: float
            Total reward of the episode.
        info: dict
            The info returned by the environment.'''
        self.episode_high_tiles[instance].append(info['high_tile'])
        self.episode_rewards[instance].append(reward)
        self.episode_scores[instance].append(info['score'])
        self.episode_steps[instance].append(info['steps'])

    def save(self, directory="logger_data", override=False):
        if not os.path.exists(directory):
            os.mkdir(directory)
        elif not override:
            return
        
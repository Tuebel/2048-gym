from gym_2048.wrappers import OneChannel
from huskarl.agent import A2C
from huskarl.policy import Greedy, EpsGreedy
from huskarl.simulation import Simulation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from validity_2048 import check_valid_2048
import gym
import gym_2048
import matplotlib.pyplot as plt
import numpy as np


def create_env():
    env = gym.make('2048-4x4-v0')
    return OneChannel(env)


dummy_env = create_env()
model = Sequential([
    Flatten(input_shape=dummy_env.observation_space.shape),
    Dense(4000, activation='relu'),
    Dense(2000, activation='relu'),
    Dense(4, activation='linear')
])

# We will be running multiple concurrent environment instances
instances = 8
eps_max = 0.2
# Create a policy for each instance with a different distribution for epsilon
policy = [Greedy(check_valid_2048)] + [
    EpsGreedy(eps, check_valid_2048) for eps in np.arange(
        0, eps_max, eps_max/(instances-1))]

# Create Advantage Actor-Critic agent
agent = A2C(model, actions=dummy_env.action_space.n, nsteps=2,
            instances=instances, policy=policy,
            test_policy=Greedy(check_valid_2048))


def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('A2C rewards')
    # for i, (ed, steps) in enumerate(zip(episode_rewards, episode_steps)):
    #     plt.plot(steps, ed, alpha=0.5 if i == 0 else 0.2,
    #              linewidth=2 if i == 0 else 1)
    plt.plot(episode_steps[0], episode_rewards[0])
    plt.show() if done else plt.pause(0.0001)


# Create simulation, train and then test
print('training the agent')
sim = Simulation(create_env, agent)
sim.train(max_steps=100000, instances=instances, plot=plot_rewards)
print('testing')
sim.test(max_steps=1000)
model.save('a2c_agent_v0.tf')

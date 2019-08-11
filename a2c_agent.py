import gym
import gym_2048
import matplotlib.pyplot as plt
import numpy as np
from huskarl.agent import A2C
from eps_greedy_2048 import EpsGreedy2048
from huskarl.simulation import Simulation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten


def create_env(): return gym.make('2048-4x4-v0').unwrapped


dummy_env = create_env()
model = Sequential([
    Flatten(input_shape=dummy_env.observation_space.shape),
    Dense(200, activation='relu'),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(20, activation='relu')
])


# We will be running multiple concurrent environment instances
instances = 8

# Create a policy for each instance with a different distribution for epsilon
policy = [EpsGreedy2048(0)] + [EpsGreedy2048(eps)
                               for eps in np.arange(0, 1, 1/(instances-1))]

# Create Advantage Actor-Critic agent
agent = A2C(model, actions=dummy_env.action_space.n,
            nsteps=2, instances=instances, policy=policy)


def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('A2C rewards')
    for i, (ed, steps) in enumerate(zip(episode_rewards, episode_steps)):
        plt.plot(steps, ed, alpha=0.5 if i == 0 else 0.2,
                 linewidth=2 if i == 0 else 1)
    plt.show() if done else plt.pause(0.0001)


# Create simulation, train and then test
print('training the agent')
sim = Simulation(create_env, agent)
sim.train(max_steps=100000, instances=instances, plot=plot_rewards)
print('testing')
sim.test(max_steps=1000)
model.save('a2c_agent_v0.tf')

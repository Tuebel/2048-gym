import gym
import gym_2048
import huskarl as hk
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten


def create_env(): return gym.make('2048-3x3-v0').unwrapped


dummy_env = create_env()
model = Sequential([
    Flatten(input_shape=dummy_env.observation_space.shape),
    Dense(20, activation='relu'),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(20, activation='relu')
])
# Create Deep Q-Learning Network agent
agent = hk.agent.DQN(model, actions=dummy_env.action_space.n, nsteps=2)


def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
    # Pause a bit so that the graph is updated
    plt.show() if done else plt.pause(0.001)


# Create simulation, train and then test
print('training the agent')
sim = hk.Simulation(create_env, agent)
sim.train(max_steps=10000, visualize=False, plot=plot_rewards)
print('testing policy')
sim.test(max_steps=1000)

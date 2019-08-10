import gym
import gym_2048
from huskarl.agent import DQN
from huskarl.policy import EpsGreedy
from huskarl.simulation import Simulation
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten


def create_env(): return gym.make('2048-4x4-v1').unwrapped


dummy_env = create_env()
model = Sequential([
    Flatten(input_shape=dummy_env.observation_space.shape),
    Dense(200, activation='relu'),
    Dense(200, activation='relu'),
    Dense(50, activation='relu')
])
# Create Deep Q-Learning Network agent
agent = DQN(model, actions=dummy_env.action_space.n, gamma=0.99,
            batch_size=128, nsteps=2, enable_double_dqn=True)


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
sim = Simulation(create_env, agent)
# explore and reduce epsilon
agent.policy = EpsGreedy(0.2)
sim.train(max_steps=10000, visualize=False, plot=plot_rewards)
agent.policy = EpsGreedy(0.1)
sim.train(max_steps=10000, visualize=False, plot=plot_rewards)
agent.policy = EpsGreedy(0.05)
sim.train(max_steps=10000, visualize=False, plot=plot_rewards)
agent.policy = EpsGreedy(0.01)
sim.train(max_steps=10000, visualize=False, plot=plot_rewards)
print('testing policy')
sim.test_policy = EpsGreedy(0),
sim.test(max_steps=1000)
model.save('dqn_agent_log_two.tf')

from gym_2048.wrappers import ClipReward, OneChannel
from huskarl.agent import A2C
from huskarl.policy import Greedy, EpsGreedy
from huskarl.simulation import Simulation
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from validity_2048 import check_valid_2048
import gym
import gym_2048
import matplotlib.pyplot as plt
import numpy as np


def create_env():
    env = gym.make('2048-4x4-v0')
    # Clip rewards as they explode in longer games.
    # Long running games should still lead to high rewards
    env = ClipReward(env)
    # Wrap for Conv2D
    return OneChannel(env)


def plot_rewards(episode_rewards, episode_steps, done=False, do_show=False):
    '''Plot the rewards of each episode'''
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('A2C rewards')
    # for i, (ed, steps) in enumerate(zip(episode_rewards, episode_steps)):
    #     plt.plot(steps, ed, alpha=0.5 if i == 0 else 0.2,
    #              linewidth=2 if i == 0 else 1)
    plt.plot(episode_steps[0], episode_rewards[0])
    if done and do_show:
        plt.show()
    else:
        plt.pause(0.001)


def run_epoch(create_env, agent, max_steps=50000, max_test_steps=500,
              do_show=False):
    '''Runs one epoch of training and testing'''
    def plot_epoch(rewards, steps, done=False): plot_rewards(
        rewards, steps, done=done, do_show=do_show)
    # train the instance
    sim = Simulation(create_env, agent)
    sim.train(max_steps=max_steps, instances=instances, plot=plot_rewards)
    # test
    sim.test(max_steps=max_test_steps)
    # TODO modify huskarl to analyze the observation: highscore, avg


# Create model

dummy_env = create_env()
initializer = VarianceScaling()
model = Sequential([
    Conv2D(15, 3, activation='relu', padding='same',
           input_shape=dummy_env.observation_space.shape,
           kernel_initializer=initializer),
    MaxPool2D(pool_size=(2, 2), strides=None,
              padding='valid', data_format=None),
    Conv2D(15, 2, activation='relu', padding='valid',
           input_shape=dummy_env.observation_space.shape,
           kernel_initializer=initializer),
    Flatten(),
    Dense(500, activation='relu', kernel_initializer=initializer),
    Dense(500, activation='relu', kernel_initializer=initializer),
])
# Optimizer with sheduled learning rate decay
optimizer = Adam(lr=3e-3, decay=1e-5)
# Run multiple instances
instances = 8
# Exploration and learning rate decay after each epoch
eps_max = 0.3
eps_decay = 0.9
learning_rate = 3e-3
learning_decay = 0.9
# Create Advantage Actor-Critic agent
agent = A2C(model, actions=dummy_env.action_space.n, nsteps=2,
            instances=instances, optimizer=optimizer,
            test_policy=Greedy(check_valid_2048))

# Run epochs
for epoch in range(20):
    # Create a policy for each instance with a different eps
    policy = [Greedy(check_valid_2048)] + [
        EpsGreedy(eps, check_valid_2048) for eps in np.arange(
            0, eps_max, eps_max/(instances-1))]
    # Update agent
    agent.policy = policy
    agent.model.optimizer.lr = learning_rate
    # Run epoch
    run_epoch(create_env, agent, max_steps=10000, max_test_steps=500,
              do_show=False)
    # Decay
    eps_max *= eps_decay
    learning_rate *= learning_decay
    plt.savefig(f'epoch_{epoch}.png')
# Show plot of final epoch
run_epoch(create_env, agent, max_steps=5000, max_test_steps=500,
          do_show=True)
model.save('a2c_epoch_agent_v0.tf')

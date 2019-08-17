from gym_2048.wrappers import ClipReward, OneChannel
from huskarl.agent import A2C
from huskarl.policy import Greedy, EpsGreedy
from huskarl.simulation import Simulation
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
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


def plot_rewards(episode_rewards, episode_steps, done=False, do_show=False,
                 title='A2C rewards'):
    '''Plot the rewards of each episode'''
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(title)
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
    # TODO move epochs to own class
    def plot_epoch(rewards, steps, done=False): plot_rewards(
        rewards, steps, done=done, do_show=do_show)
    # train the instance
    sim = Simulation(create_env, agent)
    print('training')
    sim.train(max_steps=max_steps, instances=instances, max_subprocesses=0,
              plot=plot_rewards)
    # test
    print('testing')
    sim.test(max_steps=max_test_steps, visualize=False)


# Create model
#   - Convolutions: spatial observations
#   - Dropout: catastrophic forgetting (high numbers, what were low one?)
#     https://arxiv.org/pdf/1312.6211.pdf

dummy_env = create_env()
initializer = VarianceScaling()
model = Sequential([
    Conv2D(8, 4, activation='elu', padding='same',
           input_shape=dummy_env.observation_space.shape,
           kernel_initializer=initializer),
    Conv2D(16, 2, activation='elu', padding='valid',
           input_shape=dummy_env.observation_space.shape,
           kernel_initializer=initializer),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='elu', kernel_initializer=initializer)
])
# Optimizer with sheduled learning rate decay
optimizer = Adam(lr=3e-3, decay=1e-5)
# Run multiple instances
instances = 8
# Exploration and learning rate decay after each epoch
eps_max = 0.2
eps_decay = 0.9
learning_rate = 3e-3
learning_decay = 0.9
# Create Advantage Actor-Critic agent
agent = A2C(model, actions=dummy_env.action_space.n, nsteps=20,
            instances=instances, optimizer=optimizer,
            test_policy=Greedy(gym_2048.check_valid))

# Run epochs
for epoch in range(20):
    # Create a policy for each instance with a different eps
    policy = [Greedy(gym_2048.check_valid)] + [
        EpsGreedy(eps, gym_2048.check_valid) for eps in np.arange(
            0, eps_max, eps_max/(instances-1))]
    # Update agent
    agent.policy = policy
    agent.model.optimizer.lr = learning_rate
    # Run epoch
    print(f'Epoch {epoch}')
    run_epoch(create_env, agent, max_steps=10000, max_test_steps=500,
              do_show=False)
    # Decay
    eps_max *= eps_decay
    learning_rate *= learning_decay
    # logging
    plt.savefig(f'a2c_epoch_{epoch}.png')
# Show plot of final epoch
run_epoch(create_env, agent, max_steps=5000, max_test_steps=500,
          do_show=True)
model.save('a2c_epoch_agent_v0.tf')

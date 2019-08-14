from gym_2048.wrappers import ClipReward, OneChannel
from huskarl.agent import DQN
from huskarl.policy import Greedy, EpsGreedy
from huskarl.simulation import Simulation
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from validity_2048 import check_valid_2048
import gym
import gym_2048
import matplotlib.pyplot as plt


def create_env():
    '''Returns a custom environment for the agent'''
    env = gym.make('2048-4x4-v0')
    env = ClipReward(env)
    return OneChannel(env)


dummy_env = create_env()
initializer = VarianceScaling()
model = Sequential([
    Conv2D(10, 3, activation='relu', padding='same',
           input_shape=dummy_env.observation_space.shape,
           kernel_initializer=initializer),
    MaxPool2D(pool_size=(2, 2), padding='valid'),
    Flatten(),
    Dropout(0.5),
    Dense(500, activation='relu', kernel_initializer=initializer),
    Dropout(0.5),
    Dense(500, activation='relu', kernel_initializer=initializer)
])

# Exploration and learning rate decay after each epoch
eps = 0.2
eps_decay = 0.9
learning_rate = 3e-3
learning_decay = 0.9
explore_policy = EpsGreedy(eps, check_valid_2048)
exploit_policy = Greedy(check_valid_2048)
# Create Deep Q-Learning Network agent
agent = DQN(model, actions=dummy_env.action_space.n, gamma=0.99,
            batch_size=64, nsteps=2, enable_double_dqn=True,
            enable_dueling_network=True, target_update=10,
            test_policy=exploit_policy)


def plot_rewards_show(episode_rewards, episode_steps, done=False,
                      title='Rewards'):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('DQN rewards')
    for i, (ed, steps) in enumerate(zip(episode_rewards, episode_steps)):
        plt.plot(steps, ed, alpha=0.5 if i == 0 else 0.2,
                 linewidth=2 if i == 0 else 1)
    plt.show() if done else plt.pause(0.0001)


def plot_rewards(episode_rewards, episode_steps, done=False,
                 title='Rewards'):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('DQN rewards')
    plt.plot(episode_steps[0], episode_rewards[0])
    plt.pause(0.0001)


# Create simulation and run epochs
print('training the agent')
sim = Simulation(create_env, agent)
for epoch in range(20):
    print(f"Epoch {epoch}")
    # Update agent
    explore_policy.eps = eps
    agent.model.optimizer.lr = learning_rate
    # explore then exploit
    agent.policy = explore_policy
    sim.train(max_steps=20000, visualize=False, plot=plot_rewards)
    agent.policy = exploit_policy
    sim.train(max_steps=2000, visualize=False, plot=plot_rewards)
    explore_policy.eps *= eps_decay
    # Decay
    eps *= eps_decay
    learning_rate *= learning_decay
    # logging
    plt.savefig(f'dqn_epoch_{epoch}.png')

agent.policy = exploit_policy
sim.train(max_steps=2000, visualize=False, plot=plot_rewards_show)
print('testing policy')
sim.test(max_steps=1000)
model.save('dqn_agent_v0.tf')

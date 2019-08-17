from gym_2048.wrappers import ClipReward, OneChannel
from huskarl.agent import DQN
from huskarl.policy import Greedy, EpsGreedy
from huskarl.simulation import Simulation
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
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

# Exploration and learning rate decay after each epoch
eps = 0.2
eps_decay = 0.9
learning_rate = 3e-3
learning_decay = 0.9
explore_policy = EpsGreedy(eps, gym_2048.check_valid)
exploit_policy = Greedy(gym_2048.check_valid)
# Create Deep Q-Learning Network agent
agent = DQN(model, actions=dummy_env.action_space.n, gamma=0.99,
            batch_size=64, nsteps=50, enable_double_dqn=True,
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
    plt.plot(episode_steps[0], episode_rewards[0], '#8A2BE2')
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
    # logging
    plt.savefig(f'dqn_epoch_{epoch}.png')
    sim.train(max_steps=2000, visualize=False, plot=plot_rewards)
    explore_policy.eps *= eps_decay
    # Decay
    eps *= eps_decay
    learning_rate *= learning_decay

agent.policy = exploit_policy
sim.train(max_steps=2000, visualize=False, plot=plot_rewards_show,
          log_info=lambda info: print(info))
print('testing policy')
sim.test(max_steps=1000)
model.save('dqn_agent_v0.tf')

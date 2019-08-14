from gym_2048.wrappers import LogObservation, OneChannel
from huskarl.agent import DQN
from huskarl.policy import Greedy, EpsGreedy
from huskarl.simulation import Simulation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from validity_2048 import check_valid_2048
import gym
import gym_2048
import matplotlib.pyplot as plt


def create_env():
    '''Returns a custom environment for the agent'''
    env = gym.make('2048-4x4-v0')
    env = LogObservation(env)
    return OneChannel(env)


dummy_env = create_env()
model = Sequential([
    Conv2D(15, 3, activation='relu', padding='valid',
           input_shape=dummy_env.observation_space.shape),
    Flatten(),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(4, activation='linear')
])

# Create a policy for each instance with a different distribution for epsilon
explore_policy = EpsGreedy(0.1, check_valid_2048)
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
    for i, (ed, steps) in enumerate(zip(episode_rewards, episode_steps)):
        plt.plot(steps, ed, alpha=0.5 if i == 0 else 0.2,
                 linewidth=2 if i == 0 else 1)
    plt.pause(0.0001)


# Create simulation
print('training the agent')
sim = Simulation(create_env, agent)
# switch between exploitation and exploration
for i in range(50):
    agent.policy = exploit_policy
    sim.train(max_steps=2000, visualize=False, plot=plot_rewards)
    agent.policy = explore_policy
    sim.train(max_steps=2000, visualize=False, plot=plot_rewards)
agent.policy = exploit_policy
sim.train(max_steps=2000, visualize=False, plot=plot_rewards_show)
print('testing policy')
sim.test(max_steps=1000)
model.save('dqn_agent_v0.tf')

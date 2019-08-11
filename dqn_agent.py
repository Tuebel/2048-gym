from huskarl.agent import DQN
from huskarl.policy import Greedy, EpsGreedy
from huskarl.simulation import Simulation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from validity_2048 import check_valid_2048
import gym
import gym_2048
import matplotlib.pyplot as plt


def create_env(): return gym.make('2048-4x4-v0').unwrapped


dummy_env = create_env()
model = Sequential([
    Flatten(input_shape=dummy_env.observation_space.shape),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu')
])
# Create Deep Q-Learning Network agent
agent = DQN(model, actions=dummy_env.action_space.n, gamma=0.99,
            batch_size=64, nsteps=2, enable_double_dqn=True,
            enable_dueling_network=True, target_update=100,
            test_policy=Greedy(check_valid_2048))


def plot_rewards(episode_rewards, episode_steps, done=False,
                 title="Rewards"):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(title)
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
    # Pause a bit so that the graph is updated
    # plt.show() if done else plt.pause(0.0001)
    plt.pause(0.0001)


# Create simulation, train and then test
print('training the agent')
sim = Simulation(create_env, agent)
# explore and reduce epsilon
n_eps = 2
eps_max = 0.1
eps_min = 0
for i in range(n_eps):
    eps = eps_max - (eps_max-eps_min) / (n_eps - 1) * i
    agent.policy = EpsGreedy(eps, check_valid_2048)

    def plot_wrapper(episode_rewards, episode_steps, done=False):
        return plot_rewards(episode_rewards, episode_steps, done,
                            title=f'Rewards for eps={eps}')

    sim.train(max_steps=100, visualize=False,
              plot=plot_wrapper)

print('testing policy')
sim.test(max_steps=1000)
model.save('dqn_agent_v0.tf')

import gym
import gym_2048

env = gym.make('2048-3x3-v0')
env.reset()
for i_episode in range(20):
    observation = env.reset()
    for i_action in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        if done:
            print(f'Episode finished after {i_action} actions.\n'
                  f'Final score {reward}\n')
            break
env.close()

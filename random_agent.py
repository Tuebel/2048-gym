import gym
import gym_2048

env = gym.make('2048-3x3-v0')
env.reset()
highscore = 0
for i in range(10000):
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    if reward > highscore:
        highscore = reward
    if done:
        env.reset()
        print(f'Final score {reward}\n')
env.close()
print(f'Highscore {highscore}\n')

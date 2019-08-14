import gym
import gym_2048
from gym_2048.wrappers import ClipReward

env = gym.make('2048-4x4-v0')
env = ClipReward(env)
env.reset()
score = 0
highscore = 0
average = 0
n_episodes = 200
steps = 0
avg_steps = 0
for i in range(n_episodes):
    finished = False
    while not finished:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        steps += 1
        score += reward
        finished |= done
    if score > highscore:
        highscore = score
    average += score / n_episodes
    print(f'Final score {score}\n')
    env.reset()
    score = 0
env.close()
avg_steps = steps / n_episodes
print(f'Highscore {highscore}\nAverage {average}\nSteps {steps}'
      f' average {avg_steps}')

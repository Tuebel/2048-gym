import gym
import gym_2048

env = gym.make('2048-4x4-v0')
env.reset()
score = 0
highscore = 0
average = 0
n_episodes = 200
for i in range(n_episodes):
    finished = False
    while not finished:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        score += reward
        finished |= done
    if score > highscore:
        highscore = score
    average += score / n_episodes
    print(f'Final score {score}\n')
    env.reset()
    score = 0
env.close()
print(f'Highscore {highscore}\n Average {average}')
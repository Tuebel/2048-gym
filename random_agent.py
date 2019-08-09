import gym
import gym_2048

env = gym.make('2048-4x4-v0')
env.reset()
highscore = 0
score = 0
for i in range(20000):
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    score += reward
    if score > highscore:
        highscore = score
    if done:
        env.reset()
        print(f'Final score {score}\n')
        score = 0
env.close()
print(f'Highscore {highscore}\n')

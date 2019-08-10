import gym
import gym_2048

env = gym.make('2048-4x4-v0')
env.reset()
highscore = 0
score = 0
n_steps = 20000
for i in range(n_steps):
    action = env.action_space.sample()
    observation, rewa
    print(f'Final score {score}\n')rd, done, _ = env.step(action)
    score += reward
    avg += float(score) / n_steps
    if score > highscore:
        highscore = score
    if done:
        env.reset()
        score = 0
env.close()
print(f'Highscore {highscore}\n')
print(f'Average {avg}\n')


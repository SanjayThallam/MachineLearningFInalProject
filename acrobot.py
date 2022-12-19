import gymnasium as gym
import os
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

env_randomGuess  = gym.make("Acrobot-v1", render_mode="human")
allEpisodes_randomGuess = []
allScores_randomGuess = []
episodes = 0
for episode in range(1, episodes+1):
    state = env_randomGuess.reset()
    done = False
    score = 0

    while not done:
        env_randomGuess.render()
        action = env_randomGuess.action_space.sample()
        obs, reward, done, info, extra = env_randomGuess.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    allEpisodes_randomGuess.append(episode)
    allScores_randomGuess.append(score)
env_randomGuess.close()
del env_randomGuess

log_path = os.path.join('Training', 'Logs')

env = make_vec_env("Acrobot-v1", n_envs=1)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps = 200000)

print(evaluate_policy(model, env, n_eval_episodes=20, render=True))
env.close()
allEpisodes = []
allScores = []
episodes = 20
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
    allEpisodes.append(episode)
    allScores.append(score[0])
env.close()

print("Random Guess")
print(allEpisodes_randomGuess)
print(allScores_randomGuess)

print("Testing")
print(allEpisodes)
print(allScores)
"""
plt.plot(allEpisodes_randomGuess, allScores_randomGuess)
plt.xlim([0, max(allEpisodes_randomGuess)])
plt.ylim([min(allScores_randomGuess) - 100, max(allScores_randomGuess) + 100])
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Random Action")
plt.show()
"""

plt.plot(allEpisodes, allScores)
plt.xlim([0, max(allEpisodes)])
plt.ylim([min(allScores) - 50, max(allScores) + 100])
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("MLP")
plt.show()

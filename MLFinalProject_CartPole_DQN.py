from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from matplotlib import pyplot as plt
import gymnasium as gym
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


env_randomGuess = gym.make("CartPole-v1", render_mode="human")
allEpisodes_randomGuess = []
allScores_randomGuess = []

episodes = 10
for episode in range(1, episodes+1):
    state = env_randomGuess.reset()
    done = False
    score = 0

    while not done:
        env_randomGuess.render()
        action = env_randomGuess.action_space.sample()
        obs, reward, done, info, extra = env_randomGuess.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
    allEpisodes_randomGuess.append(episode)
    allScores_randomGuess.append(score)
env_randomGuess.close()
del env_randomGuess

env = make_vec_env("CartPole-v1", n_envs=1)

from stable_baselines3 import DQN
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300000)

print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
env.close()

allEpisodes = []
allScores = []

episodes = 10
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

plt.plot(allEpisodes_randomGuess, allScores_randomGuess)
plt.xlim([0, max(allEpisodes_randomGuess)])
plt.ylim([0, max(allScores_randomGuess) + 100])
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Random Action")
plt.show()


plt.plot(allEpisodes, allScores)
plt.xlim([0, max(allEpisodes)])
plt.ylim([0, max(allScores) + 100])
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("DQN")
plt.show()

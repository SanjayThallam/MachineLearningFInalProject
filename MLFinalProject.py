import gymnasium as gym
import os
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


# env = gym.make("CartPole-v1", render_mode="human")
allEpisodes = []
allScores = []

# episodes = 10
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         obs, reward, done, info, extra = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
#     allEpisodes.append(episode)
#     allScores.append(score)
# env.close()
# print(allEpisodes)
# print(allScores)
# plt.plot(allEpisodes, allScores)
# # plt.xlim([0, max(allEpisodes)])
# # plt.ylim([0, max(allScores) + 100])
# # plt.title("Random Action Scores vs Episode")
# plt.show()


log_path = os.path.join('Training', 'Logs')

env = make_vec_env("CartPole-v1", n_envs=1)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
env.close()

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
    allScores.append(score)
env.close()

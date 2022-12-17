##Tried getting Breakout game to work, could not get it to work fully
import gymnasium as gym
import os
from matplotlib import pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory


env = gym.make('ALE/Breakout-v5', render_mode="human")
states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info, extra = env.step(action)
        score+=reward

    print('Episode:{} Score:{}'.format(episode, score))


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.add(Flatten())
    return model

model = build_model(states, actions)

print(model.summary())


def build_agent(model, actions):
    policy = BoltzmannGumbelQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)

    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)


scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

_ = dqn.test(env, nb_episodes=5, visualize=True)

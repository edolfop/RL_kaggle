import random
import time

import numpy as np
import gym
import matplotlib.pyplot as plt

from IPython.display import clear_output

############Functions#####################

def run(env, num_episodes=50):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = env.action_space.sample()  # take a random action
            state, reward, done, info = env.step(action)[:4]
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return rewards

def display_human(env):
    env.reset()
    rgb_array = env.render()
    print(rgb_array)
    #time.sleep(1) 

def random_run(env):
    while True:
        action = env.action_space.sample()  # take a random action
        state, reward, done, info, a = env.step(action)
        if done:
            break
    return reward

def run(env, num_episodes=50):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = env.action_space.sample()  # take a random action
            state, reward, done, info, a = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return rewards



##########CODE##################
env_name = 'FrozenLake-v1'

env = gym.make(env_name , is_slippery=False, render_mode='human')

print(np.__version__)
print("Antes del display")
display_human(env)
print("Despues del display")

#Actions
# 0 LEFT
# 1 DOWN
# 2 RIGHT
# 3 UP

random_run(env)

env.close()






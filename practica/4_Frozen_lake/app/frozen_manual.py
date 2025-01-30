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



##########CODE##################
env_name = 'FrozenLake-v1'
#env = gym.make(env_name , render_mode= 'rgb_array')
env = gym.make(env_name , is_slippery=False, render_mode='human')
#env.render()
print(np.__version__)
print("Antes del display")
display_human(env)
print("Despues del display")
#Actions
# 0 LEFT
# 1 DOWN
# 2 RIGHT
# 3 UP
action = 2
state, reward, done, info, a = env.step(action)
print("Despues del step")
print('State: ', state, ' Reward: ', reward, ' Done: ', done, ' Info: ', info)

action = 2
state, reward, done, info, a = env.step(action)
print("Despues del step")
print('State: ', state, ' Reward: ', reward, ' Done: ', done, ' Info: ', info)

action = 1
state, reward, done, info, a = env.step(action)
print("Despues del step")
print('State: ', state, ' Reward: ', reward, ' Done: ', done, ' Info: ', info)

action = 1
state, reward, done, info, a = env.step(action)
print("Despues del step")
print('State: ', state, ' Reward: ', reward, ' Done: ', done, ' Info: ', info)

action = 1
state, reward, done, info, a = env.step(action)
print("Despues del step")
print('State: ', state, ' Reward: ', reward, ' Done: ', done, ' Info: ', info)

action = 2
state, reward, done, info, a = env.step(action)
print("Despues del step")
print('State: ', state, ' Reward: ', reward, ' Done: ', done, ' Info: ', info)

#display_human(env)

# informations about the environment
'''print("Environment:", env_name)
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Reward Range:", env.reward_range)
print("Number of Actions:", env.action_space.n)
if hasattr(env, 'get_action_meanings'):
    print("Action Meanings:", env.get_action_meanings())
if hasattr(env, 'get_keys_to_action'):
    print("Keys to Action:", env.get_keys_to_action())'''

env.close()






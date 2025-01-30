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

def q_learning(env, num_episodes=500, alpha=0.07, gamma=0.99, epsilon=0.3):
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    print("Action Space: ", action_space_size, " State Space:", state_space_size)
    q_table = np.zeros((state_space_size, action_space_size))
    rewards = [] # store rewards and number of actions taken
    num_actions_list = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        num_actions_episode = 0

        while True:
            # epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # explore
            else:
                action = np.argmax(Q[state])  # exploit
            next_state, reward, done, _ = env.step(action)[:4]
            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action]) # q-value update ny the q-learning formula

            state = next_state
            total_reward += reward
            num_actions_episode += 1
            if done:
                break

        print("Episode: ", episode , "Q table: " , q_table)
        rewards.append(total_reward)
        num_actions_list.append(num_actions_episode)
    
    print("Q-table after training:")
    print(q_table) 

    np.save('q_table.npy', q_table)
    return rewards, num_actions_list



##########CODE##################
env_name = 'FrozenLake-v1'

env = gym.make(env_name , is_slippery=False, render_mode='human')
#Actions
# 0 LEFT
# 1 DOWN
# 2 RIGHT
# 3 UP

print(np.__version__)
print("Antes del display")
display_human(env)
print("Despues del display")

q_learning(env)


#display_human(env)

env.close()






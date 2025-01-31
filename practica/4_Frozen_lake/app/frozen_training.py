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

# mia
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
            next_state, reward, done, _ = env.step(action)
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

# kaggle
def q_training (env):
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    print("Action Space: ", action_space_size, " State Space:", state_space_size)
    q_table = np.zeros((state_space_size, action_space_size))
    num_episodes = int(1e3)
    max_steps_per_episode = 30

    # lerning rate
    lr = 1e-1
    # gamma
    dr = 0.99

    # exp rate por todo el entrenamiento
    exploration_rate = 1
    max_exp_rate = 1
    min_exp_rate = 1e-3
    exp_decay_rate = 1e-3

    # building a reward list for all the episodes
    rewards_all_episodes = []

    # Q-learning algorithm
    for episode in range(num_episodes):
        # reset the movements in the env
        state, _ = env.reset()
        # check if the agent reaches the target
        done = False
        
        # variable for expected return G_t
        rewards_current_episode = 0
        
        print("Episode: ", episode)
        # for loop for each step for the agent
        for step in range(max_steps_per_episode):
            
            # apply epsilon greedy stategy
            random_number = random.uniform(0, 1)
            # Exploration Vs. Exploitation trade-off
            if random_number > exploration_rate:
                # start exploitation ---> getting the maximum Q-value from the possible movements of his current state.
                action = np.argmax(q_table[state, :])
            else:
                # start exploration ---> select any random action to explore a random state.
                action = env.action_space.sample()
            
            # after taking the action, we're going to update our agent with the new info, rewards, state, and if he reaches the end or not!
            #new_state, reward, done, info = env.step(action)
            new_state, reward, done, info = env.step(action)[:4]

            # Update our Q-table for Q(s, a) using Bellman Equation
                                                # Old Q-value
            q_table[state, action] = (1 - lr) * q_table[state, action] + lr * (reward + dr*(np.max(q_table[new_state, :])))
            #q_table[state[0], action] = (1 - lr) * q_table[state, action] + lr * (reward + dr*(np.max(q_table[new_state, :])))
                                                # learned value

            # transition to the next state
            state = new_state
            rewards_current_episode += reward
            
            # check to see if our last action ended the episode for us,
            # meaning, did our agent step in a hole or reach the goal?
            if done:
                break
            # If the action did end the episode, then we jump out of this loop and move on to the next episode.
            # Otherwise, we transition to the next time-step.
        
        # Exploration Rate Decay
        # https://en.wikipedia.org/wiki/Exponential_decay
        exploration_rate = min_exp_rate + \
                        (max_exp_rate - min_exp_rate) * np.exp(-exp_decay_rate * episode)
        
        # append the current rewards in the list of rewards
        rewards_all_episodes.append(rewards_current_episode)

    np.save('q_table.npy', q_table)
    return q_table


##########CODE##################
env_name = 'FrozenLake-v1'

env = gym.make(env_name , is_slippery=False, render_mode='human')
#Actions
# 0 LEFT
# 1 DOWN
# 2 RIGHT
# 3 UP

print(np.__version__)
#display_human(env)

Q_table = q_training(env)

print("Q: " , Q_table)

#display_human(env)

env.close()






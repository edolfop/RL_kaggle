#matplotlib inline
import warnings
warnings.simplefilter(action = 'ignore')
import os
import gym
import random
import imageio
import subprocess
import numpy as np
from glob import glob
from IPython import display
import IPython.display as ipd
from matplotlib import pyplot as plt
from IPython.display import clear_output 


def display(env: gym.Env) -> None:
    env.reset()
    img = plt.imshow(env.render()) 
    plt.axis('off')
    plt.show()

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



def discretize_state(state, bins):
    """Discretize the continuous state into discrete bins."""
    if isinstance(state, tuple):
        state = state[0]
    discretized_state = tuple(np.digitize(s, bins[i]) - 1 for i, s in enumerate(state))
    return discretized_state
def q_learning(env, num_episodes=500, alpha=0.07, gamma=0.99, epsilon=0.3):
    num_bins = [40, 40] # discretize the state space
    state_bins = [np.linspace(-1.2, 0.6, num_bins[0]),
                  np.linspace(-0.07, 0.07, num_bins[1])]
    print("State bins: ", state_bins)
    num_actions = env.action_space.n # initialize q-table
    Q = np.zeros((num_bins[0], num_bins[1], num_actions))
    rewards = [] # store rewards and number of actions taken
    num_actions_list = []
    for episode in range(num_episodes):
        state = discretize_state(env.reset(), state_bins)
        total_reward = 0
        num_actions_episode = 0

        while True:
            # epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # explore
            else:
                action = np.argmax(Q[state])  # exploit
            next_state, reward, done, _ = env.step(action)[:4]
            next_state = discretize_state(next_state, state_bins)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action]) # q-value update ny the q-learning formula
            state = next_state
            total_reward += reward
            num_actions_episode += 1
            if done:
                break
        os.system('cls')
        print("Episode: ", episode , " Rewards: " , total_reward, " Num. Actions: " , num_actions_episode , end="", flush=True)
        rewards.append(total_reward)
        num_actions_list.append(num_actions_episode)
    np.save('q_table.npy', Q)
    return rewards, num_actions_list

def play_game_with_frames(env, Q_table, max_steps=1000):
    output_directory = "C:/frames_play"
    os.makedirs(output_directory, exist_ok=True)
    state = env.reset()
    state = discretize_state(state, state_bins)
    print(f"Initial state: {state}")
    episode_frames = []
    total_reward = 0
    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            episode_frames.append(frame)
        action = np.argmax(Q_table[state])  # select action using q-table
        next_state, reward, done, _ = env.step(action)[:4]
        state = discretize_state(next_state, state_bins)
        total_reward += reward
        if done:
            break
    print(f"Total Reward: {total_reward}")
    for i, frame in enumerate(episode_frames): # save frames for this episode
        image_path = os.path.join(output_directory, f"frame_{i:03d}.png")
        imageio.imwrite(image_path, (frame * 255).astype(np.uint8))
    env.close()

np.set_printoptions(precision=15)  # Configura la visualización de decimales
warnings.simplefilter(action = 'ignore')

# First wehave to load the Q-table
Q_table = np.load('q_table.npy')
print(f"Q-table size: {Q_table.shape}")
print(Q_table)

# We initiate the enviroment
env = gym.make('MountainCar-v0', render_mode="rgb_array")

state_bins = [np.linspace(-1.2, 0.6, 40), np.linspace(-0.07, 0.07, 40)] # assuming we have the state_bin from training

initial_state = env.reset()
print(f"Initial state before the loop: {initial_state}")
initial_state = discretize_state(initial_state, state_bins)

# playing the game with the trained q-table for 1 episode, save frames, and print total reward
play_game_with_frames(env, Q_table)
# useing ffmpeg to create a video from saved frames
video_filename = "cassietvid.mp4"
#os.system(f"ffmpeg -framerate 30 -pattern_type glob -i 'frames/*.png' -c:v libx264 -pix_fmt yuv420p {video_filename}")
#ffmpeg -framerate 30 -i C:/frames_play/frame_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p C:/frames_play/output_video.mp4

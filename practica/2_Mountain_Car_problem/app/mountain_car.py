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

env_name = 'MountainCar-v0'
env = gym.make(env_name)
# informations about the environment
print("Environment:", env_name)
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Reward Range:", env.reward_range)
print("Number of Actions:", env.action_space.n)
if hasattr(env, 'get_action_meanings'):
    print("Action Meanings:", env.get_action_meanings())
if hasattr(env, 'get_keys_to_action'):
    print("Keys to Action:", env.get_keys_to_action())
env.close()

print(env.observation_space.low) # minimum possible values for each variable in the state space
print(env.observation_space.high) # maximum possible values for each variable in the state space


warnings.simplefilter(action = 'ignore')

env = gym.make('MountainCar-v0')
#initial_state = env.reset() # reset the environment and get the initial observation
#print("Initial State:", initial_state)
env = gym.make('MountainCar-v0', render_mode="rgb_array")
state = env.reset()

output_directory = "C:/frames"
os.makedirs(output_directory, exist_ok=True)

frames = [] # store frames
for i in range(500): # take 100 random actions
    
    # rendering the current frame using the environment's render method
    frame = env.render()
    frames.append(frame)
    
    action = env.action_space.sample() #take a random action
    state, reward, done, _ = env.step(action)[:4]
    
for i, frame in enumerate(frames):
    image_path = os.path.join(output_directory, f"frame_{i:03d}.png")
    imageio.imwrite(image_path, (frame * 255).astype(np.uint8))

framerate = 25 
# create a video from saved frames
print("Guardando Video")
#os.system(f"ffmpeg -framerate {framerate} -i {output_directory}/frame_%03d.png -c:v libx264 -r {framerate} -pix_fmt yuv420p {output_directory}/A_Mountain_500.mp4 2>NUL")
os.system("ffmpeg -framerate 25 -i C:/frames/frame_%03d.png -c:v libx264 -r 25 -pix_fmt yuv420p C:/frames/A_Mountain_500.mp4")
#ffmpeg -framerate 25 -i C:/frames/frame_%03d.png -c:v libx264 -r 25 -pix_fmt yuv420p C:/frames/output_video.mp4
print(f"{output_directory}/Mountain_500.mp4")

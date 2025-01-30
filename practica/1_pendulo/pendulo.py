import gymnasium as gym
import numpy as np
#import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import imageio

env_name = 'CartPole-v1'
env = gym.make(env_name)
# informations about the environment
print("Environment:", env_name)
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Reward Range:", env.reward_range)
print("Number of Actions:", env.action_space.n)
env.close()
'''
env = gym.make("CartPole-v1", render_mode="rgb_array")

observation, _ = env.reset()

terminated = False

print(observation)

# inicializar el video
output_directory = "C:/frames_pendulo"
os.makedirs(output_directory, exist_ok=True)
frames = []

while True:

    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    if terminated:
        break
    
    
    if observation[2] >=0:
        observation, reward, terminated, info, _ = env.step(1)
    if observation[2] <=0:
        observation, reward, terminated, info, _ = env.step(0)
    
    
    rgb_array = env.render()
    
    frames.append(rgb_array)
    
print("done")

for i, frame in enumerate(frames): # save frames for this episode
    image_path = os.path.join(output_directory, f"frame_{i:03d}.png")
    imageio.imwrite(image_path, (frame * 255).astype(np.uint8))'''

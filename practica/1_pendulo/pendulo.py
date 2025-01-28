import gymnasium as gym
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

env = gym.make("CartPole-v1", render_mode="rgb_array")

observation, _ = env.reset()

terminated = False

print(observation)

rgb_array = env.render()

plt.imshow(rgb_array)

frames = []

while True:
    
    if terminated:
        break
    
    
    if observation[2] >=0:
        observation, reward, terminated, info, _ = env.step(1)
    if observation[2] <=0:
        observation, reward, terminated, info, _ = env.step(0)
    
    
    rgb_array = env.render()
    
    frames.append(rgb_array)
    
print("done")


for i, frame in enumerate(frames):
    clear_output(wait=True)
    plt.imshow(frame)
    plt.title(f"frame: {i}")
    plt.show()
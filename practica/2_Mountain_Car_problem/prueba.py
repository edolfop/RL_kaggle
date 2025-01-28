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

env = gym.make('MountainCar-v0', render_mode="rgb_array")
state = env.reset()

output_directory = "/kaggle/working/framesrandom/"
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
os.system(f"ffmpeg -framerate {framerate} -i {output_directory}/frame_%03d.png -c:v libx264 -r {framerate} -pix_fmt yuv420p videomelhes500.mp4 2>NUL")
import os

import matplotlib
print(os.getcwd())

import gym
from car_env_2 import CarEnv
from mapping.mapping import Mapping
import time
import numpy as np
from pynput import keyboard
from matplotlib import pyplot as plt

path_ = os.getcwd()

rel_path = 'models/block.xml'
path = path_ + '/' + rel_path

gym.envs.register(
     id='car-robot-left-v1',
     entry_point='car_env:CarEnv',
     max_episode_steps=2000,
	 kwargs={'model_path': path}
)
mapping = Mapping()

t0 = time.time()






env = gym.make('Ant-v4')
# env.viewer.cam.distance = env.viewer.cam.distance * 0.1   
env.reset()
t = np.arange(0, 2, 0.01)
print(t)
v_x = np.cos(2 * t)
v_y = np.sin(2 * t)
f, ax = plt.subplots(1,1, figsize=(10,10))  
for i in range(len(t)):
    print(i)
    env.render()
    # action = mapping.get_actions(v_x[i], v_y[i])
    env.step(env.action_space.sample())
    print('here')
    print(dir(env.data))
    # ax.imshow(rgb)
    # plt.pause(0.01)


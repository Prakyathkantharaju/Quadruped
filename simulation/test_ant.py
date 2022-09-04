import numpy as np

import gym

from ant_v5 import AntEnv



gym.envs.register(
     id='Ant_v7',
     entry_point='ant_v7:AntEnv',
     max_episode_steps=5000,
)

env = gym.make("Ant_v7")
env.reset()

for i in range(50):
    act = env.action_space.sample()
    obs, reward, _, info = env.step(act)
    print('obs', obs)
    print(f"x vel {info['x_velocity']}, y {info['y_velocity']}")
    print('reward', reward)
    
    


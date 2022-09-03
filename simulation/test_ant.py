import numpy as np

import gym

from ant_v5 import AntEnv



gym.envs.register(
     id='Ant_v5',
     entry_point='ant_v5:AntEnv',
     max_episode_steps=5000,
)

env = gym.make("Ant_v5")
env.reset()

for i in range(50):
    act = env.action_space.sample()
    env.step(act)
    
    

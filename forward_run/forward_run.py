import gym
 
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG, TD3

import sys, os
import numpy as np


 # typing
import numpy.typing as npt




class controller:

    def __init__(self, PPO_path: str = "/home/prakyathkantharaju/gitfolder/personal/Quadruped/forward_run/full_model_new_cost.pkl" , 
            model_path: str = "/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/models/block.xml") -> None:
        gym.envs.register(
             id='car-robot-v1',
             entry_point='car_env_5:CarEnv',
             max_episode_steps=5000,
             kwargs={'model_path': model_path, 'id': 1}
        )

        self.env = gym.make("car-robot-v1")

        self.ppo = PPO.load(PPO_path)


    def forward(self, obs: npt.ArrayLike) -> np.ndarray:
        self.ppo.predict(obs)
        action, states = self.ppo.predict(obs)
        return action


if __name__ == "__main__":
    controller()


    

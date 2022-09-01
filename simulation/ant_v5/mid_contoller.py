import gym
import numpy as np


# this HAS to be a gym env. 
class RlController:

    def __init__(self) -> None:
        
        # Create the ant env
        self.env = gym.make("ant-v4")
        obs = self.env.reset()

    def  

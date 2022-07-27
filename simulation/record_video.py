import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
#import sys and set the path
import sys, os
import numpy as np

import warnings
warnings.filterwarnings("ignore")
from stable_baselines3.common.env_checker import check_env


# for creating video
import imageio

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)



path_ = os.getcwd()

rel_path = 'models/block.xml'
path = path_ + '/' + rel_path
rel_path_2 = 'models/block_2.xml'
path_2 = path_ + '/' + rel_path_2
rel_path_3 = 'models/block_3.xml'
path_3 = path_ + '/' + rel_path_3

# load environment
from car_env_5 import CarEnv



# wandb config
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5000,
    "env_name": "car_env_v1",
}

gym.envs.register(
     id='car-robot-v1',
     entry_point='car_env_5:CarEnv',
     max_episode_steps=5000,
     kwargs={'model_path': path, 'id': 1}
)


gym.envs.register(
     id='car-robot-v2',
     entry_point='car_env_5:CarEnv',
     max_episode_steps=5000,
     kwargs={'model_path': path_2, 'id': 2}
)


gym.envs.register(
     id='car-robot-v3',
     entry_point='car_env_5:CarEnv',
     max_episode_steps=5000,
     kwargs={'model_path': path_3, 'id': 3}
)
# env.render()





def make_env(seed=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Hopper
    """
    def _init():
        # env.reset()

        env = gym.make('car-robot-v1')
        print(f"env seed: {seed}")
        return Monitor(env , info_keywords=('reward', 'distance','episode_length', 'id'), filename=f'.run_logs/logs/{run.id}_1')

    set_random_seed(seed)
    return _init

def make_env_2(seed=0):
	"""
	Create a wrapped, monitored SubprocVecEnv for Hopper
	"""
	def _init():
		# env.reset()

		env = gym.make('car-robot-v2')
		print(f"env seed: {seed}")
		return Monitor(env, info_keywords=('reward', 'distance','episode_length', 'id'), filename=f'.run_logs/logs/{run.id}_2')

	set_random_seed(seed)
	return _init

def make_env_3(seed=0):
	"""
	Create a wrapped, monitored SubprocVecEnv for Hopper
	"""
	def _init():
		# env.reset()

		env = gym.make('car-robot-v3')
		print(f"env seed: {seed}")
		return Monitor(env, info_keywords=('reward', 'distance','episode_length', 'id'), filename=f'.run_logs/logs/{run.id}_3')

	set_random_seed(seed)
	return _init


if __name__ == '__main__':
    env_list = [make_env(0), make_env_2(1), make_env_3(2)]

    PATH = ""
    model = PPO("MlpPolicy", 'car-robot-v3', verbose=1) 
    model.load(PATH)
    model.eval()

    env = gym.make('car-robot-v3')
    obs = env.reset()
    vector = []
    for i in range(4000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        vector.append(env.render(mode='rgb_array'))
        print(info)
        if done:
            obs = env.reset()
            print("done")
            break

    writer  = imageio.get_writer(f'.run_logs/videos/v3.mp4', fps=int(1/0.01))
    for i in vector:
        writer.append_data(i)
    writer.close()




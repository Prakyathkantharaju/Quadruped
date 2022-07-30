import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
#import sys and set the path
import sys, os
import numpy as np


import matplotlib.pyplot as plt

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
        # return Monitor(env , info_keywords=('reward', 'distance','episode_length', 'id'), filename=f'.run_logs/logs/{run.id}_1')
        return env

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
        # return Monitor(env, info_keywords=('reward', 'distance','episode_length', 'id'), filename=f'.run_logs/logs/{run.id}_2')
        return env

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
        # return env
        # return Monitor(env, info_keywords=('reward', 'distance','episode_length', 'id'), filename=f'.run_logs/logs/{run.id}_3')
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_list = [make_env(0), make_env_2(1), make_env_3(2)]
    env = gym.make('car-robot-v1')
    env.reset()
    # train_env = SubprocVecEnv(env_list, start_method='fork')

    # PATH = "/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/models/1tp97qwb/model.zip"
    PATH = "/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/.other/full_model_new.pkl"
    model = PPO.load(PATH)
    # mean_reward, std_reward = evaluate_policy(model, env , n_eval_episodes=10, render=True)

    # print(f"mean reward: {mean_reward}")
    # print(f"std reward: {std_reward}")
    
    # model.eval()

    # env = make_env_2(1)         
    obs = env.reset()
    vector = []
    action_history_1 = []
    action_history_2 = []
    buffer_store = []
    fig, ax = plt.subplots(2,2)
    for i in range(4000):
        action, _states = model.predict(obs)
        action_history_1.append(action[0])
        action_history_2.append(action[1])
        obs, rewards, done, info = env.step(action)
        vector.append(env.render(mode='rgb_array',width = 300, height=300, camera_name="buddy_realsense_d435i"))
        ax[0,0].imshow(vector[i])
        ax[0,1].plot(obs, label = 'obs')
        ax[1,0].plot(action_history_1, label = 'x_dot', c ='r')
        ax[1,0].legend()
        ax[1,1].plot(action_history_2, c = 'g',  label = 'y_dot')
        ax[1,1].legend()
        # plt.legend()
        plt.pause(0.00001)
        plt.draw()
        
        w, h = fig.canvas.get_width_height()
        buffer_store.append(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3))
        # print(info)
        ax[0,1].cla()
        ax[1,0].cla()
        ax[1,1].cla()
        ax[0,0].cla()

        if done:
            obs = env.reset()
            print("done")
            # break

    writer  = imageio.get_writer(f'.run_logs/videos/v3_pov.mp4', fps=int(1/0.01))
    for i in buffer_store:
        writer.append_data(i)
    writer.close()

    # plt.plot(action_history)
    # plt.show()




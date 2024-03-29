import numpy as np
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

import warnings
warnings.filterwarnings("ignore")
from stable_baselines3.common.env_checker import check_env


from ant_v5 import AntEnv


# # load wandb
import wandb
from wandb.integration.sb3 import WandbCallback

gym.envs.register(
     id='Ant_v7',
     entry_point='ant_v7:AntEnv',
     max_episode_steps=5000,
)

# wandb config
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5000,
    "env_name": "car_env_v1",
}

run = wandb.init(
    project="Ant traning",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    name="A2C-Ant-v7-desktop",
)



def make_env(seed=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Hopper
    """
    def _init():
        # env.reset()

        env = gym.make('Ant_v7')
        print(f"env seed: {seed}")
        return Monitor(env, info_keywords=('reward_forward', 'x_position', 'y_position', 'reward_ctrl', 'cur_velocity', 'x_velocity', 'y_velocity'),
                        filename=f'.run_logs/logs/{run.id}_2')

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_list = [make_env(0), make_env(1), make_env(2), make_env(3), make_env(4)]
    # env_list = [make_env(0)]

    # check_env(env)
    # train_env = DummyVecEnv(env_list)
    train_env = SubprocVecEnv(env_list, start_method='fork')
    # train_env = VecVideoRecorder(train_env, f'./.run_logs/videos/{run.id}', record_video_trigger=lambda x: x % 100000 == 0, video_length = 5000)

    train_env.reset()

    model = A2C("MlpPolicy", train_env, tensorboard_log=f"./.run_logs/logs/{run.id}", device="cuda", normalize_advantage=True,
    create_eval_env=True, verbose=1)
    # model.load("Models_parkour_large_1")

    model.learn(total_timesteps=2.5e6, log_interval=100, callback=WandbCallback(gradient_save_freq=500000,  model_save_freq=100000,
                                    model_save_path=f"./.run_logs/models/{run.id}", verbose=2))

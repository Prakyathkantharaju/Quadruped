import numpy as np 
import gym
from sklearn.metrics import mean_squared_error

# load the PPO model
from stable_baselines3 import A2C


# load the model
from ant_v5 import AntEnv
# register the env to gym
gym.envs.register(
     id='Ant_v5',
     entry_point='ant_v5:AntEnv',
     max_episode_steps=5000,
)

model_path = '/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/.other/models/1uc616nv/model.zip'
env = gym.make("Ant_v5", render_mode = "human")
agent = A2C("MlpPolicy", env)
agent.load(model_path)


obs = env.reset()

for i in range(1000):

    obs[:2] = np.array([0.5, 0])
    trq, _ = agent.predict(obs)
    obs, reward, terminate, info = env.step(trq)
    env.render()
    vel = np.array([info['x_velocity'], info['y_velocity']])
    req_vel = np.array([0.5, 0])
    # print(mean_squared_error(req_vel, vel))
    if terminate:
        print(env.i)
        obs = env.reset()
    print(vel)
    


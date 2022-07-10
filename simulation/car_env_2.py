from cv2 import inRange
import  sys
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch import ShortTensor
np.set_printoptions(threshold=sys.maxsize)

class CarEnv(mujoco_env.MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, model_path, frame_skip=1, **kwargs):
        self.model_path = model_path
        self.frame_skip = frame_skip
        observation_space = Box(low=-10, high=10, shape=(1, 10), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._i = 0
        self.velocity_store = []
        self.distance_store = []
        self.zero_vel_coutner = 0
        self.distance = 200

        mujoco_env.MujocoEnv.__init__(
            self, self.model_path, frame_skip)



    def _get_obs(self):
        incentric = self._incentric_obs()
        return incentric


    def _incentric_obs(self):
        data = self.data.sensordata[:6]
        return data

    def _get_reward(self):
        reward = -self.distance * 0.1
        velocity = self.data.qvel[0]
        reward += velocity * 0.01
        return reward

    @property
    def _alive(self):
        # dead if distance is a bit more than initial values.
        if (self.distance > 125) or (len(self.distance_store) > 200 and (self.distance > np.mean(self.distance_store[-150:-50]))) or (len(self.velocity_store) > 20 and np.mean(self.velocity_store[-5:]) < 0.001):
            return False
        else:
            return True

    def _distance(self):
        if self.viewer is not None:
            self.viewer.cam.distance = 12.0
        data = self.render(mode = "rgb_array", width = 300, height = 300)

        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        # indentifying the target.
        _, target =  cv2,inRange(data, (0, 0, 50), (50, 50,255))
        # cv2.imshow('target', target)
        _, car =  cv2,inRange(data, (50, 50, 50), (250, 250,250))
        # cv2.imshow('car', car)
        # key = cv2.waitKey(0) & 0xFF

        car_location_x, car_location_y = np.where(car > 200)
        target_location_x, target_location_y = np.where(target > 200)
        car_location = np.array([np.mean(car_location_x), np.mean(car_location_y)])
        target_location = np.array([np.mean(target_location_x), np.mean(target_location_y)])
        #print(car_location, target_location)
        distance = np.linalg.norm(target_location - car_location)
        return distance





    def step(self, action):
        self._i += 1
        self.cur_action = action
        self.do_simulation(action, self.frame_skip)
        self.distance = self._distance()
        self.distance_store.append(self.distance)

        observations = self._get_obs()
        reward = self._get_reward()
        self.velocity_store.append(self.data.qvel[0])
        done = not self._alive
        if (self.viewer.cam is not None) and (self.viewer.cam.distance > 12):
            self.viewer.cam.distance = 12.0


        if False:
            print(self.distance, np.mean(self.distance_store[-10:]))
            print(self.data.qvel[0], np.mean(self.velocity_store[-10:]))
            print(f"{self._i} reward: {reward}, alive {self._alive}, on target {self.distance}, actions {self.cur_action}")
        if self._i > 10000:
            # reward = 1000
            print(f"{self._i} reward: {reward}, alive {self._alive}, on target {self.distance}, actions {self.cur_action}")
            done = True
        if self.distance < 10:
            reward = 1000
            done = True
            print(f"{self._i} reward: {reward}, alive {self._alive}, on target {self.distance}, actions {self.cur_action}")
        return observations, reward, done, {'reward': reward, 'isalive': self._alive, 'distance': self.distance}

    def reset_model(self):
        if self.model.cam is not None:
            self.viewer.cam.distance = 12.0
        self._i = 0
        self.velocity_store = []
        self.distance_store  = []
        self.zero_vel_coutner = 0

        mujoco_env.MujocoEnv.__init__(
            self, self.model_path, self.frame_skip)


        return self._get_obs()





if __name__ == "__main__":
    carenv = CarEnv("/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/models/distance_simple.xml")

    carenv.viewer.cam.distance = 12
    print(carenv.viewer.cam.distance)
    print(carenv.action_space)
    for i in range(10):
        print(carenv.action_space.sample())
    i = 0
    carenv.step(np.array([0, 0]))
    while True:
        i += 1

        data = carenv.render(mode = "rgb_array", width = 300, height = 300)

        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        # indentifying the target.
        _, target =  cv2,inRange(data, (0, 0, 50), (50, 50,255))
        _, car =  cv2,inRange(data, (50, 50, 50), (250, 250,250))

        car_location_x, car_location_y = np.where(car > 200)
        target_location_x, target_location_y = np.where(target > 200)
        car_location = np.array([np.mean(car_location_x), np.mean(car_location_y)])
        target_location = np.array([np.mean(target_location_x), np.mean(target_location_y)])
        #print(car_location, target_location)
        distance = np.linalg.norm(target_location - car_location)

        reward = -distance * 0.01
        velocity = carenv.data.qvel[0]
        reward += velocity * 0.01
        cv2.imshow('fulll', np.hstack((data, cv2.cvtColor(target, cv2.COLOR_GRAY2BGR), cv2.cvtColor(car, cv2.COLOR_GRAY2BGR))))



        #
        free = carenv.render(mode = "rgb_array")
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("w"):
            obs, reward, _, _ = carenv.step(np.array([0, 1]))
        elif key == ord ("a"):
            obs, reward, _, _ = carenv.step(np.array([1, 0]))
        elif key ==ord("d"):
            obs, reward, _, _ = carenv.step(np.array([-1, 0]))
        elif key ==ord("s"):
            obs, reward, _, _ = carenv.step(np.array([0, -1]))
        else:
            obs, reward, _, _ = carenv.step(np.array([0, 0]))
        if carenv._alive == False:
            break

        print(f"Reward: {reward}")
        print(f"Distance: {carenv.distance}")
        # print(f"obs {obs}")


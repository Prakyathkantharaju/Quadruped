from cv2 import inRange
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self._i = 0

        mujoco_env.MujocoEnv.__init__(
            self, self.model_path, frame_skip)

        self._distance_traveled = 0

    def _get_obs(self):
        data = self.render(mode = "rgb_array",width = 300, height=300, camera_name="buddy_realsense_d435i")
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        seg =  cv2,inRange(data, (0, 0, 50), (50, 50,255))
        short_snip = seg[1][:, 230:240] / 255
        short_snip = np.sum(short_snip, axis = 1)[150:-70] / 10
        self.short_snip = short_snip
        return short_snip

    def _get_reward(self):
        reward = 0
        # reward -= np.sum(np.sqrt(self.cur_action**2)) * 0.01
        reward += self._on_target * 0.001
        reward += self.data.qvel[1]
        return reward

    @property
    def _alive(self):
        distance_traveled_las = np.all(self.data.sensordata[:2] < 1e-2)
    
        # print(f"distance traveled: {distance_traveled_las}")
        if distance_traveled_las and self._i > 200:
            return False
        else:
            return True

    @property
    def _on_target(self):
        if (np.sum(self.short_snip) > 2):
            return True
        else:
            return False

    def step(self, action):
        self._i += 1
        self.cur_action = action
        self.do_simulation(action, self.frame_skip)
        excentric_observation = self._get_obs()
        reward = self._get_reward()
        done = not self._alive
        if done:
            print(f"{self._i} reward: {reward}, alive {self._alive}, on target {self._on_target}, actions {self.cur_action}")
        if self._i > 1000:
            print(f"{self._i} reward: {reward}, alive {self._alive}, on target {self._on_target}, actions {self.cur_action}")
            done = True
        return excentric_observation, reward, done, {'reward': reward, 'isalive': self._alive, 'ontarget': self._on_target}

    def reset_model(self):
        self._i = 0
        mujoco_env.MujocoEnv.__init__(
            self, self.model_path, self.frame_skip)


        return self._get_obs()





if __name__ == "__main__":
    carenv = CarEnv("/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/models/block.xml")

    i = 0
    carenv.step(np.array([0, 0]))
    while True:
        i += 1

        data = carenv.render(mode = "rgb_array",width = 300, height=300, camera_name="buddy_realsense_d435i")
        print(f"{i}. On target {carenv._on_target}")
        print(f"{carenv.short_snip}")
        print(f"{carenv._alive}")
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        seg =  cv2,inRange(data, (0, 0, 50), (50, 50,255))
        seg_1 = seg[1]
        short_snip = seg[1][:, 230:240] / 255
        short_snip = np.sum(short_snip, axis = 1)[150:-70]
        cv2.imshow('data', data)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("w"):
            carenv.step(np.array([0, 1]))
        else:
            carenv.step(np.array([0, 0]))
        if i > 200:
            carenv.reset()

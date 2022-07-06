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
        observation_space = Box(low=-10, high=10, shape=(1, 10), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        mujoco_env.MujocoEnv.__init__(
            self, model_path, frame_skip)


    def _get_obs(self):
        data = self.render(mode = "rgb_array",width = 300, height=300, camera_name="buddy_realsense_d435i")

        # print(i)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        cv2.imshow("data", data)
        seg =  cv2,inRange(data, (0, 0, 50), (50, 50,255))
        self.seg = seg[1]
        short_snip = seg[1][:, 230:240] / 255
        short_snip = np.sum(short_snip, axis = 1)[150:-70]
        return short_snip

    def _get_reward(self):
        reward = 0
        
        reward += self._alive
        reward += self._on_target
        reward += self.sim.data.qvel[1]


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        return np.array([0, 0, 0, 0]), 0, False, {}

    def reset_model(self):
        return 0
    

if __name__ == "__main__":
    carenv = CarEnv("/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/models/block.xml")

    while True:
        carenv.step(np.array([0, 1]))
        data = carenv.render(mode = "rgb_array",width = 300, height=300, camera_name="buddy_realsense_d435i")
        # print(i)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        cv2.imshow("data", data)
        seg =  cv2,inRange(data, (0, 0, 50), (50, 50,255))
        seg_1 = seg[1]
        short_snip = seg[1][:, 230:240] / 255
        short_snip = np.sum(short_snip, axis = 1)[150:-70]


        cv2.imshow("seg", seg_1)
        k = cv2.waitKey()

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break


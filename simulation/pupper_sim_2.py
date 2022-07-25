from cv2 import inRange
import  sys
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# from mapping.mapping import Mapping

np.set_printoptions(threshold=sys.maxsize)
DEFAULT_CAMERA_CONFIG = {
    "distance": 12.0,
}
class QuadEnv(mujoco_env.MujocoEnv):
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

    def __init__(self, model_path,id = 1, frame_skip=1, weights = [0.1, 1, 0.001, 0.1], **kwargs):
        self.id = id
        self.model_path = model_path
        self.frame_skip = frame_skip
        self._i = 0
        self.velocity_store = []
        self.reward_store = []
        self.zero_vel_coutner = 0
        qpos = [0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        qvel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ctrl = [-0.12295051,  0.4353823,  -1.09235373,  0.12295051,  0.4353823,  -1.09235373,
                -0.12295051,  0.4353823,  -1.09235373,  0.12295051,  0.4353823,  -1.09235373]
        self.ctrl = ctrl
        self.init_qpos = qpos
        self.init_qvel = qvel
        # self.mapping = Mapping()
        self.weights = weights


        mujoco_env.MujocoEnv.__init__(
            self, self.model_path, frame_skip, mujoco_bindings="mujoco_py")



    def reset_model(self):
        return np.array([0])
    

    def step(self, action):
        return np.array([0]),0,0,0

    @property
    def get_state(self):
        return self.sim.get_state()



from cv2 import inRange
import  sys
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from torch import ShortTensor

from mapping.mapping import Mapping

np.set_printoptions(threshold=sys.maxsize)
DEFAULT_CAMERA_CONFIG = {
    "distance": 12.0,
}
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
        self.init_qpos = qpos
        self.init_qvel = qvel
        self.mapping = Mapping()
        self.weights = weights
        

        mujoco_env.MujocoEnv.__init__(
            self, self.model_path, frame_skip)

        

    def _get_obs(self):
        excentric = self._excentric_obs()
        # incentric = self._incentric_obs()
        # return np.concatenate([excentric, incentric])
        return excentric



    def _excentric_obs(self):
        data = self.render(mode = "rgb_array",width = 300, height=300, camera_name="buddy_realsense_d435i")
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        seg =  cv2.inRange(data, (0, 0, 50), (50, 50,255))
        short_snip = seg[220:280, 70:230]

        # cv2.imshow('wtf', short_snip)
        # cv2.rectangle(data,  (70, 225), (230, 280), (0, 0, 255), 2)
        # cv2.imshow("snip", data)
        # cv2.imshow("seg", np.mean(short_snip, axis=0))
        # print(short_snip.shape)
        short_snip = np.mean(short_snip, axis = 0)
        short_snip  =  np.split(short_snip, 20)
        short_snip = [np.mean(c) for c in short_snip]
        # short_snip = np.mean(short_snip, axis = 0)
        # print(short_snip)
        self.short_snip = short_snip
        short_snip = np.array([1 if c > 10 else 0 for c in short_snip])
        return short_snip

    def _incentric_obs(self):
        data = self.data.sensordata[:6]
        return data

    def _get_reward(self):
        reward = 0
        # alive reward
        reward += self.weights[0] * self._alive
        distance_traveled = np.copy(self.data.xpos[1]) - self.prev_position
        # forward velocity reward
        reward += self.weights[1] * self._on_target * distance_traveled[0] / 0.01
        # lateral velocity cost
        reward -= self.weights[2] * self._on_target * distance_traveled[1] / 0.01
        # action cost
        reward -= self.weights[3] * np.sqrt(self.cur_action[0] ** 2 + self.cur_action[1] ** 2)
        self.reward_store.append(reward)
        return reward



    @property
    def _alive(self):
        distance_traveled = np.copy(self.data.xpos[1]) - self.start_position
        if distance_traveled[0] < -0.05 or (not self._on_target) or (len(self.velocity_store) > 100 and np.mean(np.abs(self.velocity_store[-100:])) < 0.02):
            return False
        else: 
            return True

    

    @property
    def _on_target(self):
        if (np.sum(self.short_snip) > 100):
            return True
        else:
            return False

    def _set_action_space(self):
        self.action_space = Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)


    def step(self, action):
        if self._i < 2:
            self.start_position = np.copy(self.data.xpos[1])
            self.prev_position = 0
            actual_position = np.array([0, 0 , 0])
        else: 
            actual_position = np.copy(self.data.xpos[1]) - self.start_position
            # print(actual_position)
        self._i += 1
        self.cur_action = action
        
        throtle, steering = self.mapping.get_actions(action[0], action[1], actual_position)
        action = np.array([steering, throtle])
        self.do_simulation(action, self.frame_skip)
        excentric_observation = self._get_obs()
        reward = self._get_reward()
        done = not self._alive 
        self.velocity_store.append(self.data.qvel[0])
        self.prev_position = np.copy(self.data.xpos[1])
        
        if done:
            print(f"{self._i} reward: {sum(self.reward_store)}, alive {self._alive}, on target {self._on_target}, actions {self.cur_action}, 'id': {self.id}")
        if self._i > 5000:
            # reward = 1000
            print(f"{self._i} reward: {sum(self.reward_store)}, alive {self._alive}, on target {self._on_target}, actions {self.cur_action}, 'id': {self.id}")
            done = True

        return excentric_observation, reward, done, {'reward': reward, 'isalive': self._alive, 'ontarget': self._on_target, 'episode_length': self._i, 'id': self.id, 'distance' :actual_position[0]}

    def reset_model(self):
        self._i = 0
        self.velocity_store = []
        self.reward_store = []
        self.zero_vel_coutner = 0

        self.set_state(self.init_qpos, self.init_qvel)


        return self._get_obs()
    
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)



def euler_from_quaternion(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x,y,z,w = quat[0], quat[1], quat[2], quat[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
    
        return roll_x, pitch_y, yaw_z # in radians


if __name__ == "__main__":
    carenv = CarEnv("/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/models/block.xml")
    carenv.viewer.cam.distance = carenv.viewer.cam.distance * 0.3   
    
    i = 0
    # carenv.step(np.array([0, 0]))
    mapping = Mapping()
    start_position = np.copy(carenv.data.xpos[1])
    start_angle = euler_from_quaternion(np.copy(carenv.data.xquat[1]))


    while True:
        i += 1
        free = carenv.render("rgb_array")
        cv2.imshow("free", free)
        obs, _, _, _ = carenv.step(np.array([4, 2]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


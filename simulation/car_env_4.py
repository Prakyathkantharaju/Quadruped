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
    "distance": 20.0,
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

    def __init__(self, model_path, frame_skip=1, **kwargs):
        self.model_path = model_path
        self.frame_skip = frame_skip
        # observation_space = Box(low=-10, high=10, shape=(1, 10), dtype=np.float32)
        # self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._i = 0
        self.velocity_store = []
        self.reward_store = []
        self.prev_position = np.array([0, 0 , 0])
        self.zero_vel_coutner = 0
        qpos = [0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        qvel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.init_qpos = qpos
        self.init_qvel = qvel
        self.mapping = Mapping()
        

        mujoco_env.MujocoEnv.__init__(
            self, self.model_path, frame_skip)

        

    def _get_obs(self):
        # excentric = self._excentric_obs()

        incentric = self._incentric_obs()
        return incentric


    def _incentric_obs(self):
        data = self.data.sensordata[:9]
        return data

    def _get_reward(self):
        reward = 0
        # reward += self._on_target * 0.1
        reward +=  self.data.qvel[1] * 0.1
        actual_position = np.copy(self.data.xpos[1]) - self.start_position
        distance_traveled = actual_position[0] - self.prev_position[0]
        reward += distance_traveled
        # print(reward)
        reward -= np.sqrt(self.cur_action[0] ** 2 + self.cur_action[1] ** 2) * 0.0001
        self.reward_store.append(reward)
        return reward



    @property
    def _alive(self):
        distance_traveled = np.copy(self.data.xpos[1]) - self.start_position
        if distance_traveled[0] < -0.2 or (len(self.velocity_store) > 50 and np.mean(np.abs(self.velocity_store[-25:])) < 0.02):
            return False
        else: 
            return True

    

    def _set_action_space(self):
        self.action_space = Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)


    def step(self, action):
        if self._i < 2:
            self.start_position = np.copy(self.data.xpos[1])
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
        self.prev_position = actual_position
        
        if done:
            print(f"{self._i} reward: {sum(self.reward_store)}, alive {self._alive}, distance {actual_position[0]}, actions {self.cur_action}")
        if self._i > 5000:
            # reward = 1000
            print(f"{self._i} reward: {sum(self.reward_store)}, alive {self._alive}, distance {actual_position[0]}, actions {self.cur_action}")
            done = True

        return excentric_observation, reward, done, {'reward': reward, 'isalive': self._alive, 'distance' :actual_position[0], 'episode_length': self._i}

    def reset_model(self):
        self._i = 0
        self.velocity_store = []
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
    carenv = CarEnv("/home/prakyathkantharaju/gitfolder/personal/Quadruped/simulation/models/distance_simple_2.xml")
    # carenv.viewer.cam.distance = carenv.viewer.cam.distance * 0.3   
    
    i = 0
    # carenv.step(np.array([0, 0]))
    mapping = Mapping()
    start_position = np.copy(carenv.data.xpos[1])
    start_angle = euler_from_quaternion(np.copy(carenv.data.xquat[1]))


    while True:
        i += 1
        free = carenv.render("rgb_array")
        cv2.imshow("free", free)
        _, reward, _, _ = carenv.step(np.array([4, 0]))
        print(sum(carenv.reward_store))
        # print(carenv.data.sensordata[:9])

        # # print(free)
        # actual_position = np.array(carenv.data.xpos[1]) - np.array(start_position)
        # # if i < 200:
        # #     speed,steering = mapping.get_actions(1, 1,actual_position)
        # # elif i < 300:
        # #     speed,steering = mapping.get_actions(1, -1,actual_position)
        # # elif i < 400:
        # #     speed,steering = mapping.get_actions(1, 0,actual_position)
        # # elif i < 500:
        # #     speed,steering = mapping.get_actions(0, 0,actual_position)
        # steering = 0
        # speed = 0
        # print(f"{i}. Actual position {actual_position} Speed {speed} Steering {steering}")
        # carenv.step(np.array([steering, speed]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if i > 500:
        #     break
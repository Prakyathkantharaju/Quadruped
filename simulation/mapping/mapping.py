import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Any


class Mapping:
    """ mapping from COM to throttle and steering and other way around
    """
    def __init__(self, action_range: np.ndarray = np.array([[-1, -1],[1, 1]]), v_thr_range: np.ndarray = np.array([-1, 1]),
                 steering_range: np.ndarray = np.array([-1, 1]), verbose: bool = False) -> None:
        """
        action_range: range of action defined in the mujoco
        v_th_range: speed range the bot can move
        steering_range: angle range of the bot
        """
        self.action_range = action_range
        self.v_thr_range = v_thr_range
        self.steering_range = steering_range
        self.verbose = verbose


        # storing the previous command so that we can set the start of next optimization
        self.old_v_thr = 0
        self.old_steer =  0

    @property
    def _old(self) -> List[float]:
        return [self.old_v_thr, self.old_steer]

    @property
    def _bounds(self) -> Tuple[Any]:
        return ([self.v_thr_range[0], self.steering_range[0]],
                [self.v_thr_range[1], self.steering_range[1]])


    def _optimize(self, v_x: float, v_y: float) -> List[Any]:
        result = least_squares(self._find_vel_theta, x0 = self._old, \
                               args = (v_x, v_y), bounds = self._bounds)
        return result.x

    def _map_to_actions(self, speed: float, steering: float) -> Tuple[Any]:
        action_speed =  speed * ( self.v_thr_range[1] - self.v_thr_range[0])/\
            (self.action[0,0] - self.action[1,0]) + self.action[0,0]
        steering_speed = steering_speed * ( self.steering_range[1] - self.v_thr_range[0]) /\
            (self.action[0,1] - self.action[1,1]) + self.action[0,1]

        return action_speed, steering_speed

    def get_actions(self, v_x: float, v_y: float):
        speed_steering = self._optimize(v_x, v_y)
        # action_0, action_1 = self._map_to_actions(speed_steering[0],
        #                                           speed_steering[1])
        self.old_v_thr = speed_steering[0]
        self.old_steer = speed_steering[1]
        return self.old_v_thr, self.old_steer


    def _find_vel_theta(self, x: np.ndarray, v_x: float, v_y: float) -> List:
        # x[0] -> throttle
        # x[1] -> steering
        return [ x[0] * np.cos(x[1]) - v_x,
                 x[0] * np.sin(x[1]) - v_y]

import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Any


class Mapping:
    """ mapping from COM to throttle and steering and other way around
    """
    def __init__(self, action_range: np.ndarray = np.array([[-1, -1],[1, 1]]), v_thr_range: np.ndarray = np.array([-0.5, 0.5]),
                 steering_range: np.ndarray = np.array([-0.5, 0.5]), verbose: bool = False) -> None:
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
        self.v_x_t  = 0
        self.v_y_t = 0
        self.w_t = 0


        self.l = 0.4
        self.l_r = 0.2



    @property
    def _old(self) -> List[float]:
        return [self.old_v_thr, self.old_steer]

    @property
    def _bounds(self) -> Tuple[Any]:
        return ([self.v_thr_range[0], self.steering_range[0]],
                [self.v_thr_range[1], self.steering_range[1]])


    def _optimize(self, v_x: np.ndarray, v_y: np.ndarray, w: float, actual_angle: float) -> List[Any]:
        result = least_squares(self._find_vel_theta, x0 = self._old, \
                               args = (v_x, v_y, w, actual_angle), bounds = self._bounds)
        return result.x



    def get_actions(self,v_x: float, v_y: float, actual_position: np.ndarray) -> Tuple[Any]:
        desired_position = actual_position + np.array([v_x, v_y, 0]) * 0.01
        if abs(actual_position[0]) > 0.001 and abs(actual_position[1]) > 0.001:
            actual_angle = np.tan(actual_position[1]/actual_position[0])
        else:
            actual_angle = 0.0

        desired_angle = np.tan(desired_position[1]/desired_position[0])
        # print(desired_position)
        # v_x, v_y, w = np.array([actual_position[0],actual_position[0] + 0.03 ]), \
        #             np.array([actual_position[1], actual_position[1] + 0.03]), np.array([actual_angle, desired_angle])
        w = (desired_angle - actual_angle) / 0.01
        speed_steering = self._optimize(v_x, v_y, w, actual_angle)

        self.old_v_thr = speed_steering[0]
        self.old_steer = speed_steering[1]

        return self.old_v_thr, self.old_steer

    def _beta(self, x):
        return np.arctan((self.l_r/self.l) * np.tan(x))


    def _find_vel_theta(self, x: np.ndarray, v_x: float, v_y: float, w_x: float, actual_angle: float) -> List:
        # x[0] -> throttle
        # x[1] -> steering
        return [ x[0] * np.cos(x[1] + actual_angle) * 0.01 - v_x,
                 x[0] * np.sin(x[1] + actual_angle) * 0.01 - v_y,
                 x[0] * np.cos(self._beta(x[1])) * np.tan(x[1]) * 0.01 - w_x]

    # def _find_vel_theta(self, x: np.ndarray, v_x: np.ndarray, v_y: np.ndarray, w_x: np.ndarray) -> List:
    #     # x[0] -> throttle
    #     # x[1] -> steering
    #     # print(v_x, v_y, w_x, - v_x[1] + v_x[0])
    #     print([ x[0] * np.cos(x[1] + w_x[0]) * 0.01 - v_x[1] + v_x[0],
    #              x[0] * np.sin(x[1] + w_x[0]) * 0.01 - v_y[1] + v_y[0],
    #              x[0] * np.cos(self._beta(x[1])) * np.tan(x[1]) * 0.01 - w_x[1] + w_x[0]])

    #     return [ x[0] * np.cos(x[1] + w_x[0]) * 0.01 - v_x[1] + v_x[0],
    #              x[0] * np.sin(x[1] + w_x[0]) * 0.01 - v_y[1] + v_y[0],
    #              x[0] * np.cos(self._beta(x[1])) * np.tan(x[1]) * 0.01 - w_x[1] + w_x[0]]

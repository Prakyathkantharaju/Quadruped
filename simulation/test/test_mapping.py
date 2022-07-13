

import numpy as np
from  scipy.optimize import fsolve, least_squares
from typing import List


def find_vel_theta(x: np.ndarray, v_x: float, v_y: float) -> List:
    # x[0] -> throttle
    # x[1] -> steering
    return [ x[0] * np.cos(x[1]) - v_x,
             x[0] * np.sin(x[1]) - v_y]



root = fsolve(find_vel_theta, [0,0], (0.1, 0))
print(root)
root = least_squares(find_vel_theta, x0 = [0,0], args = (0.1, 0) \
                     , bounds = ([-1, -1], [1, 1]))

print(root)

from UDPComms import Publisher
from UDPComms import Scope


import numpy as np
import time

def direction_helper(trigger, opt1, opt2):
    if trigger == opt1:
        return -1
    if trigger == opt2:
        return 1
    return 0

def direction_helper(opt1, opt2):
    if opt1:
        return -1
    if opt2:
        return 1
    return 0


class keyboard_control:
    def __init__(self) -> None:
        self.pub = Publisher(8830, Scope.BROADCAST)
        self.MESSAGE_RATE = 20
        self.rx_ = 0.0
        self.ry_ = 0.0
        self.lx_ = 0.0
        self.ly_ = 0.0
        self.l_alpha = 0.15
        self.r_alpha = 0.3
        self._reset_msg()


    def _create_msg(self):
        self.msg = {
            "ly": 0,
            "lx": 0,
            "rx": 0,
            "ry": 0,
            "L2": 0,
            "R2": 0,
            "R1": 0,
            "L1": 0,
            "dpady": 0,
            "dpadx": 0,
            "x": 0,
            "square": 0,
            "circle": 0,
            "triangle": 0,
            "message_rate": self.MESSAGE_RATE
            }

    def _reset_msg(self):
        self._create_msg()
        self.pub.send(self.msg)
        #time.sleep(int(1 / self.MESSAGE_RATE))

    def send_start(self):
        self._create_msg()
        self.msg["L1"] = 1
        self.pub.send(self.msg)
        #time.sleep(int(1 / self.MESSAGE_RATE))

    def start_trot(self):
        self._create_msg()
        self.msg["R1"] = 1
        self.pub.send(self.msg)
        #time.sleep(int(1 / self.MESSAGE_RATE))

    def send_no_command(self):
        self._reset_msg()


    def send_controller(self, xdot: float, ydot: float) -> None:
        print(f"sending the {xdot}, {ydot}")
        # ONGOING change the mulitplication factory the receiving code
        self.msg['ly'] = np.clip(xdot, -0.5, +0.5)
        self.msg['lx'] = np.clip(ydot, -0.5, +0.5)
        self.pub.send(self.msg)
        #time.sleep(int(1 / self.MESSAGE_RATE))


if __name__ == "__main__":

    keyboard = keyboard_control() 
    keyboard.send_start()
    keyboard.send_controller(0, 0)


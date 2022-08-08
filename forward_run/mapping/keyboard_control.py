from UDPComms import Publisher
from UDPComms import Scope
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

    def send_start(self):
        self.msg['L1'] = 1
        self.pub.send(self.msg)
        time.sleep(int(1000 / self.MESSAGE_RATE))

    def send_controller(self, xdot: float, ydot: float) -> None:
        # TODO Check the X and Y velocity in the controller
        # TODO change the mulitplication factory the receiving code
        self.msg['ly'] = xdot
        self.msg['lx'] = ydot
        self.pub.send(self.msg)
        time.sleep(int(1000 / self.MESSAGE_RATE))


if __name__ == "__main__":

    keyboard = keyboard_control() 
    keyboard.send_start()
    keyboard.send_controller(0, 0)


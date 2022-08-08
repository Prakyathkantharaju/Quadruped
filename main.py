
import cv2

# Main controller to get the commands
from forward_run.forward_run import controller

# keyboard control
from forward_run.mapping.keyboard_control import keyboard_control

# main sensor
from perception.main_perception import perception


class Control_robot:
    def __init__(self) -> None:
        # main perception
        self.sensor = perception()

        # controller
        self.controller = controller()

        # keyboard
        self.keyboard = keyboard_control()

    def main(self):
        while True:
            obs = self.sensor.get_obs()

            action = self.controller.forward(obs)
            print(obs, action)
            k = cv2.waitKey(1)
            
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break





if __name__ == "__main__":
    robot = Control_robot()
    robot.main()  

    

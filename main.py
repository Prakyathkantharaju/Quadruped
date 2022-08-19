
import cv2

# Main controller to get the commands
from forward_run.forward_run import controller

# keyboard control
from forward_run.mapping.keyboard_control import keyboard_control

# main sensor
from perception.main_perception import perception


#TODO have a logger for ths recording the error and safety compaints.

class Control_robot:
    def __init__(self) -> None:
        # main perception
        self.sensor = perception()

        # controller
        self.controller = controller()

        # keyboard
        self.keyboard = keyboard_control()


        # start_ controller
        self.START = False

    def main(self):
        while True:
            obs = self.sensor.get_obs()

            action = self.controller.forward(obs)
            print(obs, action)
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            if k == ord('q'):
                print("initializing the robot")
                self.keyboard.send_start()

            if k == ord('e'):
                print("starting the trot gait")
                self.keyboard.start_trot()

                
            if k == ord('s'):
                # stopping and starting the controller
                self.START = not self.START

            if self.START:
                if sum(obs) < 1:
                    # safety
                    pass
                else:
                    self.keyboard.send_controller(action[0], action[1])

            

if __name__ == "__main__":
    robot = Control_robot()
    robot.main()
 

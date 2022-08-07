import logging

import numpy as np 
import cv2

# for typing / hinting
from typing import Tuple
import numpy.typing as npt


# for automatic color filtering
import webcolors


class perception:
    """
    main perception code reading camera and converting it into the image array
    """
    def __init__(self, position: Tuple[int, int, int, int] = (100, 190, 440, 100), n_obs: int = 20, \
            live_window: bool = True, video_capture_device: int = 0, color : str = "red"):

        """ Main perception code to read the image and convert it to the images.

        Args:
        position (Tuple[int, int, int, int]): Position of the camera in the frame. default = [300, 300, 200, 200]
        n_obs (int, optional): Number of observations as the output from the image. Defaults to 20.
        live_window (bool, optional): optional display of the live window. Defaults to True.
        video_capture_device (int, optional): optional input for the video_capture_device, Defaults to 0.
        color (str, optional): optional input for the color to segmented, Defaults to red.
        """
        self.N_OBS = n_obs

        self.CAMERA_POSITION = position

        self.LIVE_WINDOW = live_window

        self.COLOR = color

        # Creating the video capture device using camera id
        self._create_videocapture(video_capture_device)
        
        # if the live_window is selected create the windows
        if self.LIVE_WINDOW:
            self._create_live_window(name="live_window")


    def _create_videocapture(self, video_capture_device : int):
        """
        Create the capture device
         video_capture_device: int = 0
        """
        self.vid = cv2.VideoCapture(video_capture_device)


    def _create_live_window(self, name: str):
        cv2.namedWindow(name)
        self.WINDOW_NAME = name


    def get_obs(self) -> np.ndarray:
        ret, frame = self.vid.read()

        if not ret:
            logging.error("Camera not found restart the controller.")
            return np.ones_list(self.n_obs) * -1

        else:
            obs = self._process_frame(frame)
            if self.LIVE_WINDOW:
                self._display_window(frame)
            return obs

    def _process_frame(self, frame: npt.ArrayLike) -> np.ndarray:

        # converting the images into black and white
        frame_bnw = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # get color threshold value
        color = webcolors.name_to_rgb(self.COLOR)
        color = np.array(list(color))
        seg = cv2.inRange(frame_bnw, (100, 0, 0), (255, 80, 80))



        seg = seg[self.CAMERA_POSITION[1]:self.CAMERA_POSITION[1] +self.CAMERA_POSITION[3],
                self.CAMERA_POSITION[0]:self.CAMERA_POSITION[0] + self.CAMERA_POSITION[2]]
        mean_value = np.mean(seg, axis=0)
        mean_value = np.split(mean_value, self.N_OBS)

        mean_obs = [np.mean(c) for c in mean_value]

        obs = [1 if c > 10 else 0 for c in mean_obs]
        return np.array(obs)


    def _display_window(self, frame: npt.ArrayLike) -> None:
        frame_1 = cv2.rectangle(frame, (self.CAMERA_POSITION[0], self.CAMERA_POSITION[1]),
                (self.CAMERA_POSITION[0] + self.CAMERA_POSITION[2], self.CAMERA_POSITION[1] + self.CAMERA_POSITION[3]), (255, 0, 0), 2)
        cv2.imshow(self.WINDOW_NAME, frame_1)


if __name__ == "__main__":
   main = perception() 
   while True:
        obs = main.get_obs()
        print(obs)
        k = cv2.waitKey(1)
        
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")

            break


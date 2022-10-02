import os
from datetime import datetime
import numpy as np
import cv2




class RecordData:
    def __init__(self, folder: str = "record/") -> None:
        # Creating folder
        now = datetime.now()
        folder_path = folder + now.strftime("%d%m%Y-%H:%M:%S")
        os.mkdir(folder_path)

        # filepath record output
        self.output_file_path = folder_path + '/' + 'out.txt'

        # filepath record inputs
        self.input_file_path = folder_path + '/' + 'input.txt'

        # filepath record images
        self.image_folder = folder_path + '/' + 'images'
        os.mkdir(self.image_folder)
        self.input = np.array([])
        self.output = np.array([])
        self.img_n = 0



    def recordData(self, input, output, img):
        if self.img_n == 0:
            self.input = input.reshape(-1, 1)
            self.output = output.reshape(-1, 1)
            self.img_n = 1
            self.cur_img = img

        else:
            self.input = np.append(input.reshape(-1, 1), self.input, axis=1)
            self.output = np.append(output.reshape(-1, 1), self.output, axis=1)
            self.img_n += 1
            self.cur_img = img

        if self.img_n > 200:
            self._store_data()

    def _store_data(self):
        np.savetxt(self.input_file_path, self.input.T, delimiter=',')
        np.savetxt(self.output_file_path, self.output.T, delimiter=',')
        cv2.imwrite(self.image_folder + '/' + str(self.img_n) + '.jpeg', self.cur_img)

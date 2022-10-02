import cv2, time
import torch
import urllib.request
import numpy as np

import matplotlib.pyplot as plt


#model_type = "DPT_Large"
#model_type = "DPT_Hybrid"
model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/Midas", model_type)



device = torch.device("cpu")

midas.to(device)
midas.eval()



midas_transforms = torch.hub.load("intel-isl/Midas", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cam = cv2.VideoCapture(0)

plt.ion()


while True:
    start_time = time.time()
    ret, frame = cam.read()
    frame = cv2.resize(frame, (0,0), fx = 0.33, fy = 0.33)


    if not ret:
        print("failed to read anything")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed_image = transform(frame_rgb)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("closing ...")
        break

    with torch.no_grad():
        prediction = midas(transformed_image)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,).squeeze()

        output = prediction.cpu().numpy()
    end_time = time.time()
    print(end_time - start_time)

    out = cv2.normalize(output, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    combined_images = np.hstack((cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), frame))
    cv2.imshow("output", combined_images)



cam.release()

import cv2, time
import torch
import urllib.request
import pickle
import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



def construct_boxes(out:np.ndarray,x:int, y:int, n:int = 3, box_width:int = 10, thickness: int = 2):
    x_distances = x // (n + 1)  * np.arange(1, n + 1)
    y_distances = y // (n + 1)  * np.arange(1, n + 1)

    boxes = []
    for c_x in x_distances:
        for c_y in y_distances:
            boxes.append([(c_y - box_width // 2, c_x - box_width // 2),(c_y + box_width // 2, c_x + box_width // 2)])



    for box in boxes:
        cv2.rectangle(out, pt1 = box[0], pt2 = box[1], color = (0, 255, 255), thickness = thickness)
    return out, boxes

def get_distance(out:np.ndarray, boxes, verbose: True):
    results = []
    for i , box in enumerate(boxes):
        crop = out[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        results.append(np.sum(crop @ np.ones((crop.shape[0], 1))) / (255 * 9))
        if verbose:
            print(i, np.sum(crop @ np.ones((crop.shape[0], 1))) / (255 * 9))
    return results






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


print_time = False

print("#################################################")
print("############## Calibration #####################")
print("############## press C to calibrate ############")
print("############## press N to remove calibrate  ####")
print("############## press S to save calibration #####")
print("#################################################")

calibration_store = []
CALIBRATION = False

while True:
    start_time = time.time()
    ret, frame = cam.read()

    if not ret:
        print("failed to read anything")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed_image = transform(frame_rgb)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("closing ...")
        break

    if k%256 == ord("c"):
        results = get_distance(output, boxes, verbose = False)
        calibration_store.append(results)
        print("################ Stored calibration, move to next position #######")

    if k%256 == ord("n"):
        calibration_store.pop()
        print("################ Stored calibration, move to next position #######")

    if k%256 == ord("s"):
        print("############### Calibration Started ###################")
        if len(calibration_store) < 4:
            print("################ Need at least 4 datapoints ###########")
            continue
        print('$' * 25)
        print("Prepare to enter calibration distances")
        x = []
        for i in range(len(calibration_store)):
            x.append(int(input("Enter the distance 1: \n")))
        print(f"Entered values are {x}")
        x = np.array(x)
        no_boxes = len(boxes)
        x = np.repeat(x[..., np.newaxis], no_boxes, axis = 1)
        print(x)
        calibration_store_array = np.array(calibration_store)
        print(calibration_store_array.shape, x.shape)
        cal_model_store = []
        for i in range(no_boxes):
            reg = LinearRegression()
            reg.fit(calibration_store_array[:, i].reshape(-1, 1), x[:, i])
            print(f" Calibration score: {reg.score(calibration_store_array[:, i].reshape(-1,1), x[:, i])}")
            print(f" Calibration coefficient: {reg.coef_}")
            print(f" Calibration intercept: {reg.intercept_}")
            cal_model_store.append(reg)
        pickle.dump(cal_model_store, open("Calibration.pkl", "wb"))
        print("################# Saved calibration at Calibration.pkl ###############")
        CALIBRATION = True














    with torch.no_grad():
        prediction = midas(transformed_image)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,).squeeze()

        output = prediction.cpu().numpy()

    end_time = time.time()

    if print_time:
        print(end_time - start_time)

    # get the dimension of the images.
    x,y  = np.shape(output)
    # center point
    Center = [y//2, x//2]
    Boundary_width = 10
    box = [(Center[0] - 10, Center[1] -10),(Center[0] +10, Center[1] +10)]

    out = cv2.cvtColor(cv2.normalize(output, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1), cv2.COLOR_GRAY2BGR)
    out, boxes = construct_boxes(out, x, y)
    frame, _ = construct_boxes(frame, x, y)

    results = get_distance(output, boxes, verbose = False)

    #cv2.rectangle(out, pt1=box[0], pt2=box[1], color = (0, 255, 255), thickness = 3)
    #cv2.rectangle(frame, pt1=box[0], pt2=box[1], color = (0, 255, 255), thickness = 3)
    combined_images = np.hstack((out, frame))
    cv2.imshow("output", combined_images)
    if CALIBRATION:
        results = np.array(results)
        i = 0
        store_distances = []
        for value, model in zip(results, cal_model_store):
            #print(f"{i}. Current distance: {model.predict(value.reshape(1,-1))}")
            store_distances.append(model.predict(value.reshape(1,-1)).tolist()[0])
            i += 1

        print(store_distances)




cam.release()

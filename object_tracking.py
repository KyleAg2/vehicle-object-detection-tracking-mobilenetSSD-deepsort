import sys
from mobile_net import ObjectRecognition
from deep_sort.person_id_model.generate_person_features import generate_detections, init_encoder
from deep_sort.deep_sort_app import run_deep_sort, DeepSORTConfig
from deep_sort.application_util.visualization import cv2

import numpy as np

cap = cv2.VideoCapture("30 Minutes of Cars Driving By in 2009.mp4")

roi_points = []
points = 0

def printCoordinate(event, x ,y, flags, params):
    global points, roi_points

    if event==cv2.EVENT_LBUTTONDOWN:
        if points < 4:
            roi_points.append((x, y))
            points = points + 1
            print("Points: " + str(points))
        else:
            points = 0
            roi_points = []
            print("Points Reset")

model = ObjectRecognition()
encoder = init_encoder() 
config = DeepSORTConfig()

while(True):
    ret, frame = cap.read()
    boxes, centroid_list = model.get_boxes(frame, roi_points)
    
    if len(boxes) > 0:
        encoding = generate_detections(encoder, boxes, frame)
        run_deep_sort(frame, encoding, config)
        #config.time_object()
    else:
        # Display the frame without deepsort
        cv2.imshow('frame', frame) 
 
    cv2.setMouseCallback('frame', printCoordinate)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
import sys
import cv2
import numpy as np
from mobile_net import *

cap = cv2.VideoCapture("30 Minutes of Cars Driving By in 2009.mp4")

model = ObjectRecognition()

"""
def printCoordinate(event, x ,y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        #cv2.circle(frame,(x,y),3,(255,255,255),-1)
        strXY='('+str(x)+','+str(y)+')'
        print(strXY)
        #font=cv2.FONT_HERSHEY_PLAIN
        #cv2.putText(frame, strXY,(x+10, y-10),font,1,(255,255,255))
        #cv2.imshow("frame", frame)
"""

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

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_with_boxes, box_count = model.run_object_recognition(frame, roi_points)
    cv2.putText(frame_with_boxes, 'Cars: '+ str(box_count), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv2.imshow('frame', frame_with_boxes)

    cv2.setMouseCallback('frame', printCoordinate)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


'''

import sys
import cv2
import numpy as np
from mobile_net import *

image_path = "image1.jpg"

model = ObjectRecognition()

roi_points = []
points = 0

def printCoordinate(event, x ,y, flags, params):
    global points, roi_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if points < 4:
            roi_points.append((x, y))
            points = points + 1
            print("Points: " + str(points))
        else:
            points = 0
            roi_points = []
            print("Points Reset")

# Read the image
frame = cv2.imread(image_path)

# Run object recognition
frame_with_boxes, box_count = model.run_object_recognition(frame, None)
cv2.putText(frame_with_boxes, 'Cars: '+ str(box_count), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

# Display the result
cv2.imshow('frame', frame_with_boxes)
cv2.setMouseCallback('frame', printCoordinate)

# Wait for 'q' key to quit
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()



'''
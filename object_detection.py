import sys
import cv2
import numpy as np
from mobile_net import *

cap = cv2.VideoCapture("Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4")

model = ObjectRecognition()

def printCoordinate(event, x ,y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        #cv2.circle(frame,(x,y),3,(255,255,255),-1)
        strXY='('+str(x)+','+str(y)+')'
        print(strXY)
        #font=cv2.FONT_HERSHEY_PLAIN
        #cv2.putText(frame, strXY,(x+10, y-10),font,1,(255,255,255))
        #cv2.imshow("frame", frame)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_with_boxes, box_count = model.run_object_recognition(frame)
    cv2.putText(frame_with_boxes, 'Cars: '+ str(box_count), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv2.imshow('frame', frame_with_boxes)

    cv2.setMouseCallback('frame', printCoordinate)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
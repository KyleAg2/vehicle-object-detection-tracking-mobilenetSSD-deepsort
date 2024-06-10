import sys
import time
from mobile_net import ObjectRecognition
from deep_sort.person_id_model.generate_person_features import generate_detections, init_encoder
from deep_sort.deep_sort_app import run_deep_sort, DeepSORTConfig
from deep_sort.application_util.visualization import cv2

import numpy as np
#vehicle-object-detection-tracking-mobilenetSSD-deepsort\Accuracy_Test_1.mp4
#rtsp://test123:password123@192.168.254.121:554/stream2
rtsp_url = "rtsp://test123:password123@192.168.254.121:554/stream2"

cap = cv2.VideoCapture(rtsp_url)

roi_points = []
points = 0

def printCoordinate(event, x, y, flags, params):
    global points, roi_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if points < 4:
            roi_points.append((x, y))
            points += 1
            print("Points: " + str(points))
        else:
            points = 0
            roi_points = []
            print("Points Reset")

# Create a blank frame so that we don't get an error on printCoordinate
# because there is no frame at the start of the program
ret, frame = cap.read() 
if not ret:
    print("Failed to retrieve initial frame. Exiting.")
    sys.exit(1)

blank_frame = np.zeros_like(frame)

# Write "Loading... Please Wait" on the blank frame
loading_text = "Loading... Please Wait"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5  # Shorter font
font_color = (255, 255, 255)  # White color
font_thickness = 1
text_size, _ = cv2.getTextSize(loading_text, font, font_scale, font_thickness)
text_x = (blank_frame.shape[1] - text_size[0]) // 2
text_y = (blank_frame.shape[0] + text_size[1]) // 2

cv2.putText(blank_frame, loading_text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

cv2.imshow('frame', blank_frame)
cv2.waitKey(1)  # Small delay to update the window

model = ObjectRecognition()
encoder = init_encoder() 
config = DeepSORTConfig()

fps = 0
fps_counter = 0
start_time = time.time()

frame_skip = 3  # Customize this value to skip more frames (e.g., 2 means process every 2nd frame)
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Skipping iteration.")
        cap = cv2.VideoCapture(rtsp_url)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        # Skip this frame
        continue
    
    # FPS calculation
    fps_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = fps_counter / elapsed_time
        start_time = time.time()
        fps_counter = 0

    # Display FPS on the frame
    fps_text = f"FPS: {fps:.2f}"
    text_size, _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
    text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
    text_y = text_size[1] + 10  # 10 pixels from the top edge
    
    # Draw black rectangle background
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), 
                  (0, 0, 0), cv2.FILLED)
    
    # Draw the red FPS text
    cv2.putText(frame, fps_text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
    
    boxes = model.get_boxes(frame, roi_points)
    
    if len(boxes) > 0:
        encoding = generate_detections(encoder, boxes, frame)
        run_deep_sort(frame, encoding, config)
    else:
        # Display the frame without DeepSORT
        cv2.imshow('frame', frame)

    cv2.setMouseCallback('frame', printCoordinate)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

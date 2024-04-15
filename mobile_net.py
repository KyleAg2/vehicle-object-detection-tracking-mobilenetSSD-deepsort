import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

import cv2

# Maximum objects to be classified in the image
MAX_OBJECTS = 10

# Labels of interest
LABEL_SELECTOR = set([b'Car'])

#Region of interest Test
#area=[(712,440),(887,480),(494,580),(347,551),(328,497)]

def draw_region_of_interest(image, roi_points):
  if roi_points is not None: #If there is ROI, then draw, if not, skip
    cv2.polylines(image,[np.array(roi_points,np.int32)],True,(0,0,255)) #red

def detect_objects_in_ROI(point_x_of_interest, point_y_of_interest, roi_points):
  vehicle_state = cv2.pointPolygonTest(np.array(roi_points,np.int32),((point_x_of_interest,point_y_of_interest)),False) #Detect of the point is in the region of interest 
  if vehicle_state>=0: return 0 #if inside, then return 0, draw
  else: return 1  #if outside then return 1, skip

def centroid_calculation(left, right, top, bottom):
  cx=int(left+right)//2
  cy=int(bottom)
  return cx, cy

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color,
                               font, roi_points, thickness=4, display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  
  #============Calculate the Position in Bounding Box to Detect==========
  cx, cy = centroid_calculation(left, right, top, bottom)
  #============================Draw the Boxes============================
  radius = 5
  fill_color = (255, 0, 0)
  draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=fill_color)
  
  if roi_points is not None:
    if len(roi_points) >= 3:  
      if(detect_objects_in_ROI(cx, cy, roi_points)):
        return 0 #Then DONT DO ANYTHING (DRAW) BUT CONTINUE THE LOOP found in the draw_boxes method
    else: return 0

  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)
  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin



def draw_boxes(image, boxes, class_names, scores, selected_indices, roi_points, max_boxes=MAX_OBJECTS, min_score=0.3):

  """Overlay labeled boxes on an image with formatted scores and label names."""

  colors = list(ImageColor.colormap.values())
  font = ImageFont.load_default()
  box_count = 0
  for i in range(boxes.shape[0]):
    if box_count >= MAX_OBJECTS:
        break
    if i not in selected_indices:
        continue
    if scores[i] >= min_score and class_names[i] in LABEL_SELECTOR:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, roi_points, display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
      box_count += 1
  return image, box_count

def get_boxes(image, boxes, class_names, scores, selected_indices, roi_points, min_score=0.2):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  box_count = 0
  box_lst = []
  for i in range(boxes.shape[0]):
    if box_count >= MAX_OBJECTS:
        break
    if class_names[i] not in LABEL_SELECTOR or i not in selected_indices:
        continue
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      im_height, im_width, channel = image.shape
      left, right, top, bottom = (xmin * im_width, xmax * im_width,
                            ymin * im_height, ymax * im_height) 
      if roi_points is not None:
          if len(roi_points) >= 3:
            #============Calculate the Position in Bounding Box to Detect==========
            cx, cy = centroid_calculation(left, right, top, bottom)
            #============================Draw the Boxes============================
            point_of_interest = (cx, cy)
            cv2.circle(image, point_of_interest, 5, (255, 0, 0), -1)
            # Check if the box is within the ROI
            if not detect_objects_in_ROI(cx, cy, roi_points):
              box_lst.append((int(left), int(top), int(right - left), int(bottom - top)))
              box_count += 1                 
      else:
        box_lst.append((int(left), int(top), int(right - left), int(bottom - top)))
        box_count += 1     
  return np.array(box_lst)

def non_max_suppression(boxes, scores):
    selected_indices = tf.image.non_max_suppression(boxes, scores, 1000, iou_threshold=0.5,
                                                    score_threshold=float('-inf'), name=None)
    #selected_boxes = tf.gather(boxes, selected_indices)
    return selected_indices.numpy()

class ObjectRecognition:
    def __init__(self):
        module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        self.model = hub.load(module_handle).signatures['default']

    def run_object_recognition(self, frame, roi_points):
        converted_img = tf.image.convert_image_dtype(frame, tf.float32)[tf.newaxis, ...]
        result = self.model(converted_img)
        selected_indices = non_max_suppression(result['detection_boxes'], result['detection_scores'])
        result = {key: value.numpy() for key, value in result.items()}

        draw_region_of_interest(frame, roi_points) #Custom Code

        image_with_boxes, box_count = draw_boxes(
            frame, result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"], selected_indices, roi_points)
  
        return image_with_boxes, box_count

    def get_boxes(self, frame, roi_points=None):
        converted_img = tf.image.convert_image_dtype(frame, tf.float32)[tf.newaxis, ...]
        result = self.model(converted_img)
        selected_indices = non_max_suppression(result['detection_boxes'], result['detection_scores'])
        result = {key: value.numpy() for key, value in result.items()}

        draw_region_of_interest(frame, roi_points) #draw the region of interest *added

        boxes = get_boxes(
            frame, result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"], selected_indices, roi_points)
        return boxes


import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

import cv2

from roi import *

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import six

# Maximum objects to be classified in the image
MAX_OBJECTS = 15

# Labels of interest
LABEL_SELECTOR = set(['vehicle'])

#Region of interest Test
#area=[(712,440),(887,480),(494,580),(347,551),(328,497)]

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color,
                               font, roi_points, thickness=4, display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  
  #============Calculate the Position in Bounding Box to Detect==========
  cx, cy = centroid_calculation(left, right, top, bottom)
  
  if roi_points is not None:
    if len(roi_points) >= 3:  
      if not detect_objects_in_ROI(cx, cy, roi_points):
        return #Then DONT DO ANYTHING (DRAW) BUT CONTINUE THE LOOP found in the draw_boxes method
    else: return 

  #============================Draw the Boxes============================
  radius = 5
  fill_color = (255, 0, 0)
  draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=fill_color)

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



def draw_boxes(image, boxes, class_names, scores, selected_indices, roi_points, category_index, max_boxes=MAX_OBJECTS, min_score=0.5):

  """Overlay labeled boxes on an image with formatted scores and label names."""

  colors = list(ImageColor.colormap.values())
  font = ImageFont.load_default()
  box_count = 0
  for i in range(boxes.shape[0]):
    if box_count >= MAX_OBJECTS:
        break
    if i not in selected_indices:
        continue
    class_names_string = class_name_conversion(class_names[i], category_index)
    #print(class_names_string)
    if scores[i] >= min_score and class_names_string in LABEL_SELECTOR:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      #display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
      display_str = "{}: {}%".format(class_names_string, int(100 * scores[i]))
      color = colors[hash(class_names_string) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, roi_points, display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
      box_count += 1
  return image, box_count

def get_boxes(image, boxes, class_names, scores, selected_indices, roi_points, category_index, min_score=0.5):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  box_count = 0
  box_lst = []

  for i in range(boxes.shape[0]):
    if box_count >= MAX_OBJECTS:
        break
    class_names_string = class_name_conversion(class_names[i], category_index)
    if class_names_string not in LABEL_SELECTOR or i not in selected_indices:
        continue
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      im_height, im_width, channel = image.shape
      left, right, top, bottom = (xmin * im_width, xmax * im_width,
                            ymin * im_height, ymax * im_height) 
      if roi_points is not None:
          if len(roi_points) >= 3:
            #============Calculate the Position in Bounding Box to Detect============
            cx, cy = centroid_calculation(left, right, top, bottom)
            #print("Centroid Coordinates (x, y) Mobilenet:", cx, cy)

            # Check if the box is within the ROI 
            if detect_objects_in_ROI(cx, cy, roi_points):
              draw_centroid_object_tracking(cx, cy, image)
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

#=======================================================================================================

def class_name_conversion(classes, category_index):
  if classes in six.viewkeys(category_index):
    class_name = category_index[classes]['name']
  else:
    class_name = 'N/A'
  return str(class_name)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

#===========================================================================================================

class ObjectRecognition:
    def __init__(self):
        module_handle = tf.saved_model.load("saved_model(custom)v3")
        self.model = module_handle
        self.category_index = label_map_util.create_category_index_from_labelmap("Cars_label_map.pbtxt", use_display_name=True)

    def run_object_recognition(self, frame, roi_points):
        result = run_inference_for_single_image(self.model, frame)
        selected_indices = non_max_suppression(result['detection_boxes'], result['detection_scores'])
        #result = {key: value for key, value in result.items()}
        
        draw_region_of_interest(frame, roi_points) #Custom Code

        image_with_boxes, box_count = draw_boxes(
            frame, result["detection_boxes"],
            result["detection_classes"], result["detection_scores"], selected_indices, roi_points, self.category_index)
  
        return image_with_boxes, box_count

    def get_boxes(self, frame, roi_points=None):
        result = run_inference_for_single_image(self.model, frame)
        selected_indices = non_max_suppression(result['detection_boxes'], result['detection_scores'])
        #result = {key: value.numpy() for key, value in result.items()}

        draw_region_of_interest(frame, roi_points) #draw the region of interest *added

        boxes = get_boxes(
            frame, result["detection_boxes"],
            result["detection_classes"], result["detection_scores"], selected_indices, roi_points, self.category_index)
        return boxes

'''
import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

import cv2

from roi import *

# Maximum objects to be classified in the image
MAX_OBJECTS = 10

# Labels of interest
LABEL_SELECTOR = set([b'Car'])

#Region of interest Test
#area=[(712,440),(887,480),(494,580),(347,551),(328,497)]

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color,
                               font, roi_points, thickness=4, display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  
  #============Calculate the Position in Bounding Box to Detect==========
  cx, cy = centroid_calculation(left, right, top, bottom)
  
  if roi_points is not None:
    if len(roi_points) >= 3:  
      if not detect_objects_in_ROI(cx, cy, roi_points):
        return #Then DONT DO ANYTHING (DRAW) BUT CONTINUE THE LOOP found in the draw_boxes method
    else: return 

  #============================Draw the Boxes============================
  radius = 5
  fill_color = (255, 0, 0)
  draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=fill_color)

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
            #============Calculate the Position in Bounding Box to Detect============
            cx, cy = centroid_calculation(left, right, top, bottom)
            #print("Centroid Coordinates (x, y) Mobilenet:", cx, cy)

            # Check if the box is within the ROI 
            if detect_objects_in_ROI(cx, cy, roi_points):
              draw_centroid_object_tracking(cx, cy, image)
              box_lst.append((int(left), int(top), int(right - left), int(bottom - top)))
              box_count += 1                      
      else:
        box_lst.append((int(left), int(top), int(right - left), int(bottom - top)))
        box_count += 1     
  return np.array(box_lst)

def non_max_suppression(boxes, scores):
    # Ensure boxes is 2-D with shape [num_boxes, 4]
    if len(boxes.shape) == 3:
        boxes = boxes[0]  # Adjust this line based on how your data is structured
    
    # Ensure scores is 1-D with shape [num_boxes]
    if len(scores.shape) == 2:
        scores = scores[0]  # Adjust this line based on how your data is structured
    
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=1000, iou_threshold=0.5)
    return selected_indices.numpy()

class ObjectRecognition:
    def __init__(self):
        module_handle = "saved_model"
        self.model = hub.load(module_handle).signatures['serving_default']

        """
        
        module_handle = "saved_models"
        self.model = hub.load(module_handle).signatures['default']
        
        """

    def run_object_recognition(self, frame, roi_points):
        converted_img = tf.image.convert_image_dtype(frame, tf.uint8)[tf.newaxis, ...]
        result = self.model(converted_img)
        selected_indices = non_max_suppression(result['detection_boxes'], result['detection_scores'])
        result = {key: value.numpy() for key, value in result.items()}
        
        draw_region_of_interest(frame, roi_points) #Custom Code

        image_with_boxes, box_count = draw_boxes(
            frame, result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"], selected_indices, roi_points)
  
        return image_with_boxes, box_count

    def get_boxes(self, frame, roi_points=None):
        converted_img = tf.image.convert_image_dtype(frame, tf.uint8)[tf.newaxis, ...]
        result = self.model(converted_img)
        selected_indices = non_max_suppression(result['detection_boxes'], result['detection_scores'])
        result = {key: value.numpy() for key, value in result.items()}

        draw_region_of_interest(frame, roi_points) #draw the region of interest *added

        boxes = get_boxes(
            frame, result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"], selected_indices, roi_points)
        return boxes




'''

'''

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

import cv2

from roi import *

# Maximum objects to be classified in the image
MAX_OBJECTS = 10

# Labels of interest
LABEL_SELECTOR = set([b'Car'])

#Region of interest Test
#area=[(712,440),(887,480),(494,580),(347,551),(328,497)]

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color,
                               font, roi_points, thickness=4, display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  
  #============Calculate the Position in Bounding Box to Detect==========
  cx, cy = centroid_calculation(left, right, top, bottom)
  
  if roi_points is not None:
    if len(roi_points) >= 3:  
      if not detect_objects_in_ROI(cx, cy, roi_points):
        return #Then DONT DO ANYTHING (DRAW) BUT CONTINUE THE LOOP found in the draw_boxes method
    else: return 

  #============================Draw the Boxes============================
  radius = 5
  fill_color = (255, 0, 0)
  draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=fill_color)

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
      for j in range(len(class_names[i])):
          if scores[i][j] >= min_score and class_names[i][j] in LABEL_SELECTOR:
              ymin, xmin, ymax, xmax = tuple(boxes[i])
              display_str = "{}: {}%".format(class_names[i][j].decode("ascii"), int(100 * scores[i][j]))
              color = colors[hash(class_names[i][j]) % len(colors)]
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
            #============Calculate the Position in Bounding Box to Detect============
            cx, cy = centroid_calculation(left, right, top, bottom)
            #print("Centroid Coordinates (x, y) Mobilenet:", cx, cy)

            # Check if the box is within the ROI 
            if detect_objects_in_ROI(cx, cy, roi_points):
              draw_centroid_object_tracking(cx, cy, image)
              box_lst.append((int(left), int(top), int(right - left), int(bottom - top)))
              box_count += 1                      
      else:
        box_lst.append((int(left), int(top), int(right - left), int(bottom - top)))
        box_count += 1     
  return np.array(box_lst)

def non_max_suppression(boxes, scores):
    # Ensure boxes is 2-D with shape [num_boxes, 4]
    if len(boxes.shape) == 3:
        boxes = boxes[0]  # Adjust this line based on how your data is structured
    
    # Ensure scores is 1-D with shape [num_boxes]
    if len(scores.shape) == 2:
        scores = scores[0]  # Adjust this line based on how your data is structured
    
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=1000, iou_threshold=0.5)
    return selected_indices.numpy()

class ObjectRecognition:
    def __init__(self):
        module_handle = "saved_model"
        self.model = hub.load(module_handle).signatures['serving_default']

        """
        
        module_handle = "saved_models"
        self.model = hub.load(module_handle).signatures['default']
        
        """

    def run_object_recognition(self, frame, roi_points):
        converted_img = tf.image.convert_image_dtype(frame, tf.uint8)[tf.newaxis, ...]
        result = self.model(converted_img)
        selected_indices = non_max_suppression(result['detection_boxes'], result['detection_scores'])
        result = {key: value.numpy() for key, value in result.items()}
        
        draw_region_of_interest(frame, roi_points) #Custom Code

        image_with_boxes, box_count = draw_boxes(
            frame, result["detection_boxes"],
            result["detection_classes"], result["detection_scores"], selected_indices, roi_points)
  
        return image_with_boxes, box_count

    def get_boxes(self, frame, roi_points=None):
        converted_img = tf.image.convert_image_dtype(frame, tf.uint8)[tf.newaxis, ...]
        result = self.model(converted_img)
        selected_indices = non_max_suppression(result['detection_boxes'], result['detection_scores'])
        result = {key: value.numpy() for key, value in result.items()}

        draw_region_of_interest(frame, roi_points) #draw the region of interest *added

        boxes = get_boxes(
            frame, result["detection_boxes"],
            result["detection_classes"], result["detection_scores"], selected_indices, roi_points)
        return boxes

'''

#C:\Users\Technofix\Desktop\Kayle\OJT\Object Tracking with Deepsort
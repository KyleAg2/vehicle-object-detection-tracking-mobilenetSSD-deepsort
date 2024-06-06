import cv2
import numpy as np

def draw_region_of_interest(image, roi_points):

  """
  Parameters
  ----------
  image :
      image to draw
  roi_points : List[int]
      List of all points of ROI
  """

  if roi_points is not None: #If there is ROI, then draw, if not, skip
    cv2.polylines(image,[np.array(roi_points,np.int32)],True,(0,0,255)) #red

def detect_objects_in_ROI(point_x_of_interest, point_y_of_interest, roi_points):

  """
  Parameters
  ----------
  point_x_of_interest : float | int
      x coordinate of the point of interest in a object's bounding box
  point_y_of_interest : float | int
      y coordinate of the point of interest in a object's bounding box
  roi_points : List[int]
      List of all points of ROI
  """

  vehicle_state = cv2.pointPolygonTest(np.array(roi_points,np.int32),((point_x_of_interest,point_y_of_interest)),False) #Detect of the point is in the region of interest 
  if vehicle_state>=0: return 1 #if inside, then return 1, draw
  else: return 0  #if outside then return 0, skip

def centroid_calculation(left, right, top, bottom): #as long as we can get the distance between the sides, this function will work
  cx=int(left+right)//2 #center of the bounding box in x axis
  cy=int(bottom) #at the bottom of the bounding box
  return cx, cy

def draw_centroid_object_tracking(cx, cy, image):

  """
  Parameters
  ----------
  cx :
    x coordinate of the point of interest in a object's bounding box
  cy : List[int]
    y coordinate of the point of interest in a object's bounding box
  """

  point_of_interest = (cx, cy)
  cv2.circle(image, point_of_interest, 5, (255, 0, 0), -1)

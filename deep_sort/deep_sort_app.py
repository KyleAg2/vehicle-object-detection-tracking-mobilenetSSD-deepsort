from __future__ import division, print_function, absolute_import
import numpy as np
from datetime import datetime
import cv2
import firebase_admin
from firebase_admin import credentials, storage, firestore
from deep_sort.application_util import preprocessing
from deep_sort.application_util.visualization import draw_trackers
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
import tempfile
import os
import uuid

# Firebase setup
cred = credentials.Certificate('vehicle-object-detection-tracking-mobilenetSSD-deepsort/iparkpatrol-firebase-adminsdk-k8p3j-e9ac324632.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'iparkpatrol.appspot.com'  # Ensure this is your actual storage bucket
})
db = firestore.client()
bucket = storage.bucket()

def gather_sequence_info(detections, image):
    image_size = image.shape[:2]
    min_frame_idx = 1
    max_frame_idx = 1
    update_ms = 5
    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": "NA",
        "image_filenames": "NA",
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def create_detections(detection_mat, min_height=0, frame_idx=1):
    frame_indices = detection_mat[:, 0].astype(int)
    mask = frame_indices == frame_idx
    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def upload_image_to_firebase(image, track_id, unique_id): #Send things to the Firebase Storage datetime.now().isoformat()
    temp_file, temp_filename = tempfile.mkstemp(suffix='.jpg')
    try:
        cv2.imwrite(temp_filename, image)
        blob = bucket.blob(f'illegal_parking/{datetime.now().isoformat()}_{unique_id}.jpg')
        blob.upload_from_filename(temp_filename)
        blob.make_public()
        return blob.public_url
    finally:
        os.close(temp_file)
        os.remove(temp_filename)

def save_parking_info_to_firestore(track_id, image_url, position, timestamp, unique_id): #Send things to the Firebase Firestore
    doc_id = f'{timestamp.isoformat()}_{unique_id}'  # Generate a unique document ID
    doc_ref = db.collection("illegal_parking").document(doc_id)
    doc_ref.set({
        "track_id": track_id,
        "image_url": image_url,
        "position": position,
        "timestamp": timestamp.isoformat(),
        "location": "Sample Location",
        "title_of_violation": "Illegal Parking",
        "name": "",
        "address": "",
        "gender": "",
        "nationality": "",
        "licensenumber": "",
        "expiry": "",
        "restriction": "",
        "height": "",
        "weight": "",
        "platenumber": "",
        "make": "",
        "color": "",
        "model": "",
        "marking": "",
        "status": "Pending",
        "enforcerId": ""
    })

def run(image, detection, config, min_confidence,
        nms_max_overlap, min_detection_height):
    img_cpy = image.copy()
    seq_info = gather_sequence_info(detection, img_cpy)
    tracker = config.tracker

    detections = create_detections(
        seq_info["detections"], min_detection_height)
    detections = [d for d in detections if d.confidence >= min_confidence]
    
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    object_timer(tracker.tracks, config, img_cpy)

    draw_trackers(tracker.tracks, img_cpy)

def run_deep_sort(image, detection, config):
    min_confidence = 0.1
    nms_max_overlap = 1.0
    min_detection_height = 0.0
    run(image, detection, config, min_confidence, nms_max_overlap, min_detection_height)

def object_timer(tracks, config, image):
    current_time = datetime.now()
    for track in tracks:
        track_id = str(track.track_id)  # Ensure track_id is a string
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        if track_id not in config.track_start_time:
            config.track_start_time[track_id] = (current_time, track.to_tlwh())
        else:
            start_time, initial_position = config.track_start_time[track_id]
            elapsed_time = current_time - start_time
            current_position = track.to_tlwh()
            
            # Print the duration for which the vehicle has been detected
            print(f"Vehicle {track_id} timer: {elapsed_time.total_seconds():.2f} seconds")

            if np.linalg.norm(np.array(initial_position[:2]) - np.array(current_position[:2])) < 20:  # Increase position threshold
                if elapsed_time.total_seconds() > 60:
                    if track_id not in config.screenshotted_tracks:
                        print("Illegal Parking Detected:", track_id)
                        
                        # Draw bounding box around the detected vehicle
                        bbox = track.to_tlwh()
                        x, y, w, h = map(int, bbox)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color bounding box
                        unique_id = uuid.uuid4().hex  # Generate unique ID
                        image_url = upload_image_to_firebase(image, track_id, unique_id)
                        save_parking_info_to_firestore(track_id, image_url, current_position.tolist(), current_time, unique_id)
                        config.screenshotted_tracks.add(track_id)
                    del config.track_start_time[track_id]
            else:
                config.track_start_time[track_id] = (current_time, current_position)

class DeepSORTConfig:
    def __init__(self, max_cosine_distance=0.2, nn_budget=100):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        self.results = []
        self.track_start_time = {}
        self.screenshotted_tracks = set()  # Add a set to keep track of screenshotted tracks
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_mota(fp, fn, ids, gt_count):
    return 1 - (fp + fn + ids) / gt_count if gt_count != 0 else float('inf')

def calculate_motp(iou_sum, match_count):
    return iou_sum / match_count if match_count != 0 else 0

def calculate_id_switches(gt_ids, tracker_ids):
    ids = 0
    id_map = {}
    for gt_id, tracker_id in zip(gt_ids, tracker_ids):
        if gt_id in id_map:
            if id_map[gt_id] != tracker_id:
                ids += 1
        id_map[gt_id] = tracker_id
    return ids

def evaluate_tracking(ground_truth, tracker_output):
    gt_ids = ground_truth[:, 4]
    tracker_ids = tracker_output[:, 4]
    
    # Calculate IoUs and matches
    ious = np.array([calculate_iou(gt, tr)[0] for gt, tr in zip(ground_truth, tracker_output)])
    matches = ious > 0.5
    
    # Calculate FP and FN
    fp = len(tracker_ids) - np.sum(matches)
    fn = len(gt_ids) - np.sum(matches)
    
    # Calculate ID switches
    ids = calculate_id_switches(gt_ids, tracker_ids)
    
    # Calculate MOTA and MOTP
    mota = calculate_mota(fp, fn, ids, len(ground_truth))
    motp = calculate_motp(np.sum(ious[matches]), np.sum(matches))
    
    # Calculate MT and ML
    mt = np.sum(matches) / len(ground_truth)
    ml = np.sum(~matches) / len(ground_truth)
    
    return mota, motp, ids, mt, ml, fp, fn

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x1_, y1_, x2_, y2_ = box2[:4]

    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou, inter_area, union_area  # Ensure it returns a tuple

# def calculate_iou(box1, box2):
#     x1, y1, x2, y2 = map(int, box1[:4])
#     x1_, y1_, x2_, y2_ = map(int, box2[:4])

#     xi1, yi1 = max(x1, x1_), max(y1, y1_)
#     xi2, yi2 = min(x2, x2_), min(y2, y2_)

#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x2_ - x1_) * (y2_ - y1_)

#     union_area = box1_area + box2_area - inter_area

#     iou = inter_area / union_area if union_area != 0 else 0
#     return iou, inter_area, union_area

def evaluate_tracking_ds(ground_truth, tracker_output):
    gt_ids = ground_truth[:, 4]
    tracker_ids = tracker_output[:, 4]
    
    # Ensure both arrays have the same length for comparison
    min_length = min(len(gt_ids), len(tracker_ids))
    gt_ids = gt_ids[:min_length]
    tracker_ids = tracker_ids[:min_length]
    ground_truth = ground_truth[:min_length]
    tracker_output = tracker_output[:min_length]
    
    # Calculate IoUs and matches
    ious = np.array([calculate_iou(gt, tr)[0] for gt, tr in zip(ground_truth, tracker_output)])
    matches = ious > 0.5
    
    # Calculate FP and FN
    fp = len(tracker_ids) - np.sum(matches)
    fn = len(gt_ids) - np.sum(matches)
    
    # Calculate ID switches
    ids = calculate_id_switches(gt_ids, tracker_ids)
    
    # Calculate MOTA and MOTP
    mota = calculate_mota(fp, fn, ids, len(ground_truth))
    motp = calculate_motp(np.sum(ious[matches]), np.sum(matches))
    
    # Calculate MT and ML
    mt = np.sum(matches) / len(ground_truth)
    ml = np.sum(~matches) / len(ground_truth)
    
    return mota, motp, ids, mt, ml, fp, fn

from sort import *
import cv2
import numpy as np
from ultralytics import YOLO
# from google.colab.patches import cv2_imshow

# Define a single color for drawing bounding boxes (e.g., blue)
color = (255, 0, 0)

# Initialize YOLOv8 model for car detection
model = YOLO("../vehicle_detect_last.pt")
# Initialize SORT tracker
tracker = Sort()

# Paths to input and output videos on Google Drive
video_path = '../0-8.mp4'
output_path = '../output/compare_sort_2_0-8.mp4'

# Open video file
cap = cv2.VideoCapture(video_path)

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

all_ground_truths = []
all_tracker_outputs = []

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Car detection
    results = model(img, stream=True)
    detections = np.empty((0, 6))

    # Bounding boxes and confidences from YOLOv8 results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.5:
                # Since we don't have a class_id, we can set it to a default value (e.g., 0)
                class_id = 0
                currentArray = np.array([x1, y1, x2, y2, conf, class_id])
                detections = np.vstack((detections, currentArray))

    # Update tracker with detections
    resultTracker = tracker.update(detections)
    all_ground_truths.append(detections)
    all_tracker_outputs.append(resultTracker)

    # Bounding boxes and track IDs on the original frame
    for res in resultTracker:
        x1, y1, x2, y2, track_id = map(int, res[:5])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Write the frame with detections to the output video
    out.write(img)

    cv2.imshow("frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate metrics
all_ground_truths = np.vstack(all_ground_truths)
all_tracker_outputs = np.vstack(all_tracker_outputs)

mota, motp, ids, mt, ml, fp, fn = evaluate_tracking(all_ground_truths, all_tracker_outputs)
print(f"MOTA: {mota}, MOTP: {motp}, ID Switches: {ids}, Mostly Tracked: {mt}, Mostly Lost: {ml}, False Positives: {fp}, False Negatives: {fn}")
import cv2
import random
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
# from google.colab.patches import cv2_imshow

# YOLOv8 model for car detection
yolov_model = YOLO("../vehicle_detect_last.pt")

# DeepSORT tracker
tracker = DeepSort(max_age=70)

# video file
video_path = '../0-8.mp4'
cap = cv2.VideoCapture(video_path)

# video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = '../output/deep_sort_compare_2_0-8.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# window size for considering detections
window_x1, window_y1 = int(frame_width * 0.25), int(frame_height * 0.25)
window_x2, window_y2 = int(frame_width * 0.75), int(frame_height * 0.75)

# initialize variables
unique_track_ids = set()
track_colors = {}

def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

all_ground_truths = []
all_tracker_outputs = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if frame reading fails

    # car detection
    results = yolov_model(frame)
    bboxes_xywh = []
    confidences = []

    # bounding boxes and confidences from YOLOv8 results
    for result in results:
        for i, box in enumerate(result.boxes.xyxy):
            bbox = box.tolist()
            _conf = float(result.boxes.conf.cpu().numpy()[i])
            if _conf > 0.3:
                x1, y1, x2, y2 = map(int, bbox)
                w, h = x2 - x1, y2 - y1
                # only consider detections within the defined window
                if window_x1 <= x1 <= window_x2 and window_y1 <= y1 <= window_y2:
                    bboxes_xywh.append([x1, y1, w, h])
                    confidences.append(_conf)

    # Format for DeepSORT
    detections = [(bbox, conf) for bbox, conf in zip(bboxes_xywh, confidences)]

    # update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # draw the ROI window
    cv2.rectangle(frame, (window_x1, window_y1), (window_x2, window_y2), (0, 255, 0), 2)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if track_id not in track_colors:
            track_colors[track_id] = generate_random_color()

        color = track_colors[track_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Append the bounding boxes and track IDs for metric evaluation
        all_ground_truths.append([x1, y1, x2, y2, 0])  # class ID is 0 for all detections
        all_tracker_outputs.append([x1, y1, x2, y2, track_id])

    # write the frame with bounding boxes to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# calculate metrics
all_ground_truths = np.vstack(all_ground_truths)
all_tracker_outputs = np.vstack(all_tracker_outputs)

mota, motp, ids, mt, ml, fp, fn = evaluate_tracking_ds(all_ground_truths, all_tracker_outputs)
print(f"MOTA: {mota}, MOTP: {motp}, ID Switches: {ids}, Mostly Tracked: {mt}, Mostly Lost: {ml}, False Positives: {fp}, False Negatives: {fn}")

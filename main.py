import os
import json
import cv2
import numpy as np
import random
import string
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort

# === CONFIG ===
IMAGE_PATH = "input.jpg"
CROP_DIR = "cropped_birds"
SPECIES_MODEL_PATH = "bird_model.onnx"
POSE_MODEL_PATH = "yolo11n-birdpose.pt"
DETECTION_MODEL_PATH = "yolo11n.pt"
LABELS_PATH = "bird_info.json"

os.makedirs(CROP_DIR, exist_ok=True)

# === Step 1: Bird Detection (YOLO) ===
yolo_detect = YOLO(DETECTION_MODEL_PATH)
detect_results = yolo_detect([IMAGE_PATH])[0]
class_names = yolo_detect.names
bird_class_id = next(k for k, v in class_names.items() if v.lower() == "bird")

# === Step 2: Crop Detected Birds ===
image_cv = cv2.imread(IMAGE_PATH)
height, width = image_cv.shape[:2]
cropped_boxes = []
for idx, box in enumerate(detect_results.boxes):
    if int(box.cls[0]) != bird_class_id:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    cropped = image_cv[y1:y2, x1:x2]
    crop_path = os.path.join(CROP_DIR, f"bird_{idx}.jpg")
    cv2.imwrite(crop_path, cropped)
    cropped_boxes.append({
        "id": idx,
        "bird_id": ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)),
        "crop_path": crop_path,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": float(box.conf[0])
    })

# === Step 3: Species Annotation ===
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_array = json.load(f)

species_session = ort.InferenceSession(SPECIES_MODEL_PATH)
input_name = species_session.get_inputs()[0].name
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

for box in cropped_boxes:
    img = cv2.imread(box["crop_path"])
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - mean) / std
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :]
    output = species_session.run(None, {input_name: input_tensor})[0]
    pred_idx = int(np.argmax(output[0]))
    species_name = label_array[pred_idx][1]

    color_seed = hash(species_name) % 0xFFFFFF
    hex_color = "#{:02X}{:02X}{:02X}".format((color_seed >> 16) & 255, (color_seed >> 8) & 255, color_seed & 255)

    box["species"] = {
        "name": species_name,
        "color": hex_color
    }

# === Step 4: Skeleton Pose Estimation ===
KEYPOINT_NAMES = {
    0: "back", 1: "beak", 2: "belly", 3: "breast", 4: "crown", 5: "forehead",
    6: "left eye", 7: "left leg", 8: "left wing", 9: "nape",
    10: "right eye", 11: "right leg", 12: "right wing", 13: "tail", 14: "throat"
}

pose_model = YOLO(POSE_MODEL_PATH)
pose_result = pose_model([IMAGE_PATH])[0]

all_keypoints = []
if pose_result.keypoints is not None and len(pose_result.keypoints.xy) > 0:
    keypoints_xy = pose_result.keypoints.xy[0].tolist()
    keypoints_conf = pose_result.keypoints.conf[0].tolist()

    for idx, ((x, y), conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
        if conf < 0.6:
            continue
        all_keypoints.append({
            "id": idx,
            "x": float(x),
            "y": float(y),
            "confidence": float(conf)
        })

# Assign keypoints (shared for now)
for box in cropped_boxes:
    box["keypoints"] = all_keypoints

# === Step 5: Output as JSON ===
with open("final_output.json", "w", encoding="utf-8") as f:
    json.dump(cropped_boxes, f, indent=2, ensure_ascii=False)

print("✅ Done! Output written to final_output.json")

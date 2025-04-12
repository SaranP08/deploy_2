# detector.py

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os

# === CONFIG ===
MODEL_PATH = "runs/best.pt"  # Update if you moved the weights
CLASS_LABELS = {
    0: "Tear",
    1: "Unstitched",
    2: "Hole"
}

# === Load YOLOv8 model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[❌] YOLOv8 model not found at: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

def detect_damage(image, conf_threshold=0.3):
    """
    Runs YOLOv8 detection on input image.
    Returns annotated image + structured results.
    """
    results_list = []

    # Convert BGR to RGB for YOLOv8
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(rgb_img, verbose=False)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        conf = float(conf)
        cls = int(cls)

        if conf >= conf_threshold:
            label = CLASS_LABELS.get(cls, "Unknown")

            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Add to result list
            result_dict = {
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            }
            results_list.append(result_dict)

    return image, results_list


def estimate_position_cm(bbox_center, dpi=300, fabric_speed_cm_s=10, frame_index=0, fps=30):
    """
    Estimate physical position of defect based on frame index.
    Useful for video/webcam input.
    """
    time_sec = frame_index / fps
    distance_cm = fabric_speed_cm_s * time_sec
    return round(distance_cm, 2)


# === Test Block (for dev only) ===
if __name__ == "__main__":
    test_img = cv2.imread("download (3).jpg")
    if test_img is None:
        print("[❌] Couldn't read test image.")
    else:
        annotated, detections = detect_damage(test_img)
        print("[✅] Detections:", detections)
        cv2.imshow("YOLOv8 Damage Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
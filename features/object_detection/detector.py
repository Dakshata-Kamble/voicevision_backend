import cv2
from .yolo_model import get_model

model = get_model()

def detect_objects(frame):

    results = model(frame)

    detected_objects = []

    for r in results:
        for box in r.boxes:

            confidence = float(box.conf[0])

            # Ignore weak detections
            if confidence < 0.5:
                continue

            cls = int(box.cls[0])
            label = model.names[cls]

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]

            # Convert to integers
            x1 = int(x1)
            x2 = int(x2)

            # Calculate object center position
            center_x = int((x1 + x2) / 2)

            # Calculate width (used for distance estimation)
            width = int(x2 - x1)

            detected_objects.append({
                "name": label,
                "x": center_x,
                "width": width
            })

    return detected_objects
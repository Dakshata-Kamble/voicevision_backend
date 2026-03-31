from ultralytics import YOLO

# Load trained currency model
model = YOLO("models/currency_yolo.pt")

def detect_currency(frame):

    results = model(frame)

    detected_notes = []

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in detected_notes:
                detected_notes.append(label)

    return detected_notes
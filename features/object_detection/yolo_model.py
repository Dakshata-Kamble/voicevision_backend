from ultralytics import YOLO

# Load YOLO model only once when server starts
model = YOLO("yolov8n.pt")

def get_model():
    return model
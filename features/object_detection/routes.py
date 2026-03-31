from flask import Blueprint, request, jsonify
import numpy as np
import cv2
from .detector import detect_objects

# Create blueprint
object_bp = Blueprint("object_detection", __name__)

@object_bp.route("/detect-object", methods=["POST"])
def detect_object():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    objects = detect_objects(img)

    if len(objects) == 0:
        message = "No objects detected"
    else:
        names = [obj["name"] for obj in objects]
        message = "Detected: " + ", ".join(names)

    return jsonify({
        "objects": objects,
        "message": message
    })
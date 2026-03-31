from flask import Blueprint, request, jsonify
import numpy as np
import cv2
from .detector import detect_currency

currency_bp = Blueprint("currency", __name__)

@currency_bp.route("/detect-currency", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    notes = detect_currency(frame)

    if len(notes) == 0:
        message = "No currency detected"
    else:
        message = "Detected: " + ", ".join(notes)

    return jsonify({
        "notes": notes,
        "message": message
    })
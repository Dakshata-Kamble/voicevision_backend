from flask import Blueprint, request, jsonify
import numpy as np
import cv2
from .ocr_reader import read_text

text_bp = Blueprint("text_reading", __name__)

@text_bp.route("/read-text", methods=["POST"])
def read_text_api():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    text = read_text(img)

    return jsonify({
        "text": text
    })
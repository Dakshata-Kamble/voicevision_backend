from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load trained model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "currency_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (IMPORTANT: order must match training folders)
CLASS_NAMES = ["10", "100", "20", "200", "50", "500"]

@app.route("/")
def home():
    return "VoiceVision Currency Detection API Running"

@app.route("/predict-currency", methods=["POST"])
def predict_currency():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    # Convert bytes to image
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preprocess image
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_index = np.argmax(preds)
    confidence = float(np.max(preds))

    result = CLASS_NAMES[class_index]

    return jsonify({
        "denomination": result,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run()

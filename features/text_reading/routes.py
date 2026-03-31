from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import traceback # एरर ढूंढने के लिए
from .ocr_reader import perform_smart_ocr

text_bp = Blueprint('text_reading', __name__)

@text_bp.route('/read-text', methods=['POST'])
def read_text():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image found"}), 400
        
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # OCR प्रोसेस करें
        extracted_text = perform_smart_ocr(img)

        return jsonify({"text": extracted_text})

    except Exception as e:
        # अगर कोई बड़ी एरर आए, तो उसे टर्मिनल में दिखाएं और JSON भेजें
        print("Backend Error Details:")
        traceback.print_exc() 
        return jsonify({"error": str(e)}), 500
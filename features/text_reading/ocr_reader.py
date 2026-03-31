import pytesseract
import cv2
import numpy as np
import re

def perform_smart_ocr(image):
    try:
        # 1. इमेज को बड़ा और साफ़ करें
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # कॉन्ट्रास्ट बढ़ाना (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # 2. अक्षरों को मोटा करना (Dilation) ताकि OCR आसानी से पढ़ सके
        # यह कम रोशनी या दूर की फोटो के लिए बहुत कारगर है
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(gray, kernel, iterations=1)
        processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 3. OCR Configuration
        # PSM 6: पैराग्राफ को एक ब्लॉक की तरह पढ़ना (Sentence formation के लिए)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, lang="eng", config=custom_config)

        # 4. सफाई (Cleaning)
        # सिर्फ काम के शब्द रखना
        text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text).strip()
        # एक्स्ट्रा गैप हटाकर लाइन बनाना
        clean_text = " ".join(text.split())

        print(f"Detected Sentence: {clean_text}")
        return clean_text

    except Exception as e:
        print(f"OCR Processing Error: {e}")
        return ""
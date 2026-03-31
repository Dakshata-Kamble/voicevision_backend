import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def read_text(image):

    # resize strongly
    image = cv2.resize(image, None, fx=3, fy=3)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove noise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # adaptive threshold (VERY IMPORTANT)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # OCR config (single block text)
    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(thresh, config=config)

    return text.strip()
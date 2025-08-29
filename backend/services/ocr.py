# backend/services/ocr_engine.py
import logging
from typing import Tuple, List, Dict, Any
from PIL import Image, UnidentifiedImageError
import pytesseract
from pytesseract import Output
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import os
from backend.config import settings

logger = logging.getLogger("ocr_engine")

# Ensure pytesseract finds the tesseract exe if provided
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


# --------------------------
# Internal OCR runner
# --------------------------
def _ocr_sync(image: Image.Image) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Synchronous OCR using pytesseract, returns (full_text, word_rows).
    Each word row has: text, conf, left, top, width, height
    """
    try:
        text = pytesseract.image_to_string(image)
        data = pytesseract.image_to_data(image, output_type=Output.DICT)

        rows: List[Dict[str, Any]] = []
        for i in range(len(data.get("text", []))):
            txt = (data["text"][i] or "").strip()
            conf = data["conf"][i]

            if txt != "" and conf != "-1":
                try:
                    conf_val = float(conf)
                except Exception:
                    conf_val = 0.0

                rows.append({
                    "text": txt,
                    "conf": conf_val,
                    "left": int(data.get("left", [0])[i]),
                    "top": int(data.get("top", [0])[i]),
                    "width": int(data.get("width", [0])[i]),
                    "height": int(data.get("height", [0])[i]),
                })

        return text, rows
    except Exception as e:
        logger.exception("OCR failed: %s", e)
        return "", []


# --------------------------
# Threaded OCR wrapper
# --------------------------
_executor = ThreadPoolExecutor(max_workers=2)

def run_ocr(image: Image.Image) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Run OCR asynchronously in a thread pool.
    Returns: (full_text, structured_word_rows)
    """
    future = _executor.submit(_ocr_sync, image)
    return future.result()


# --------------------------
# Preprocessing helpers
# --------------------------
def _preprocess_image(cv_img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for OCR: grayscale, threshold, denoise
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold for varying lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    # Denoise (optional)
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)
    return denoised


# --------------------------
# Public OCR entrypoints
# --------------------------
def extract_text(image_path: str) -> str:
    """
    Extract plain text from any supported image (JPG, PNG, WEBP).
    """
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"Unsupported or corrupted image: {image_path}")

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    processed = _preprocess_image(cv_img)

    text = pytesseract.image_to_string(processed)
    return text


def extract_text_with_conf(image_path: str) -> Dict[str, Any]:
    """
    Extract structured OCR result with confidence + bounding boxes.
    Returns: { full_text: str, words: [ {text, conf, left, top, width, height}, ... ] }
    """
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"Unsupported or corrupted image: {image_path}")

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    processed = _preprocess_image(cv_img)

    full_text, rows = _ocr_sync(Image.fromarray(processed))
    return {"full_text": full_text, "words": rows}

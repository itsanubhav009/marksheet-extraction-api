# backend/utils/pdf_utils.py
from pdf2image import convert_from_bytes
from typing import List
from PIL import Image
import logging

logger = logging.getLogger("file_handler")

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 300, poppler_path: str | None = None) -> List[Image.Image]:
    """
    Convert PDF bytes into a list of PIL.Image objects (RGB).
    On Windows, pass poppler_path from settings if poppler not on PATH.
    """
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=poppler_path)
        return [p.convert("RGB") for p in pages]
    except Exception as e:
        logger.exception("pdf2image conversion failed")
        raise

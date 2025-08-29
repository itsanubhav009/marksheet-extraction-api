# backend/services/preprocess.py
"""
Advanced image preprocessing for OCR (Tesseract).
- Resizing (target_width)
- Optional deskew (requires OpenCV)
- Denoise (median / bilateral filtering)
- Contrast enhancement (CLAHE if OpenCV available)
- Adaptive thresholding (OpenCV if available, else Otsu-like fallback)
- Optional debug save for inspection

This file tries to use OpenCV (cv2) if installed for better quality. If not,
it falls back to Pillow + numpy methods.
"""
from typing import Optional
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import os
import io
import logging

logger = logging.getLogger("preprocess")

# Try importing OpenCV; if not available, operate without it.
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

def _pil_to_cv2(img: Image.Image) -> "np.ndarray":
    """Convert PIL Image to CV2 BGR numpy array"""
    arr = np.array(img.convert("RGB"))
    # Convert RGB -> BGR
    return arr[:, :, ::-1].copy()

def _cv2_to_pil(img: "np.ndarray") -> Image.Image:
    """Convert CV2 BGR/gray numpy array to PIL Image (RGB)"""
    if len(img.shape) == 2:  # gray
        return Image.fromarray(img)
    # BGR -> RGB
    rgb = img[:, :, ::-1]
    return Image.fromarray(rgb)

def _deskew_cv2(gray: "np.ndarray") -> "np.ndarray":
    """
    Estimate skew angle and rotate to deskew the image using cv2.
    Works on grayscale images (uint8).
    """
    coords = cv2.findNonZero(cv2.bitwise_not(gray))
    if coords is None:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    logger.debug("deskew angle: %.3f", angle)
    return rotated

def preprocess_image(
    img: Image.Image,
    target_width: Optional[int] = 1600,
    upscale_small: bool = True,
    deskew: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
    adaptive_thresh_cv2: bool = True,
    debug_save_path: Optional[str] = None,
) -> Image.Image:
    """
    Full preprocessing pipeline for OCR.
    Args:
        img: PIL.Image input
        target_width: preferred width in pixels (typical: 1400-1800). If input is smaller and upscale_small True, image is upscaled.
        deskew: whether to attempt de-skewing (uses OpenCV if available)
        denoise: whether to apply denoising (median / bilateral)
        enhance_contrast: whether to apply contrast enhancement (CLAHE if cv2 available, else Pillow Enhance)
        adaptive_thresh_cv2: prefer OpenCV adaptive threshold if cv2 available; else use Pillow+numpy fallback
        debug_save_path: if provided, saves intermediate images for debugging (folder path). saved files: pre_resize.png, gray.png, denoised.png, thresh.png
    Returns:
        PIL.Image (binarized) suitable for Tesseract OCR
    """

    # 1) Ensure RGB and optionally resize/upscale
    img = img.convert("RGB")
    orig_w, orig_h = img.size

    # If target width provided: upscale (if smaller) or downscale (if larger)
    if target_width:
        if (orig_w < target_width and upscale_small) or (orig_w > target_width):
            ratio = target_width / float(orig_w)
            new_h = int(round(orig_h * ratio))
            img = img.resize((target_width, new_h), resample=Image.LANCZOS)

    # 2) Convert to grayscale
    gray_pil = ImageOps.grayscale(img)

    # Save debug
    if debug_save_path:
        os.makedirs(debug_save_path, exist_ok=True)
        gray_pil.save(os.path.join(debug_save_path, "gray.png"))

    # 3) Option A: use OpenCV pipeline (preferred)
    if _HAS_CV2:
        # Convert to CV2 grayscale
        gray_cv = _pil_to_cv2(gray_pil)
        if len(gray_cv.shape) == 3:
            # convert RGB->GRAY if necessary
            gray_cv = cv2.cvtColor(gray_cv, cv2.COLOR_BGR2GRAY)

        # Deskew
        if deskew:
            try:
                gray_cv = _deskew_cv2(gray_cv)
            except Exception:
                logger.exception("Deskew failed; continuing without deskew")

        # Denoise: bilateral filter keeps edges while reducing noise
        if denoise:
            try:
                gray_cv = cv2.bilateralFilter(gray_cv, d=9, sigmaColor=75, sigmaSpace=75)
            except Exception:
                logger.exception("cv2 bilateralFilter failed")

        # Contrast - CLAHE
        if enhance_contrast:
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_cv = clahe.apply(gray_cv)
            except Exception:
                logger.exception("CLAHE failed")

        # Adaptive threshold (good for uneven lighting)
        thresh_img = None
        if adaptive_thresh_cv2:
            try:
                # cv2.ADAPTIVE_THRESH_GAUSSIAN_C often works well
                thresh_img = cv2.adaptiveThreshold(
                    gray_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, blockSize=15, C=10
                )
            except Exception:
                logger.exception("adaptiveThreshold failed; falling back to Otsu")
        if thresh_img is None:
            # Otsu's thresholding
            try:
                _, thresh_img = cv2.threshold(gray_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                logger.exception("Otsu threshold failed; using simple numpy threshold")
                # fallback: simple mean threshold
                arr = np.array(gray_cv)
                thr = int(np.clip(arr.mean() - 10, 50, 200))
                thresh_img = (arr > thr).astype("uint8") * 255

        # optional morphological cleanup: close small holes
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        except Exception:
            pass

        # Save debug
        if debug_save_path:
            try:
                debug_path = os.path.join(debug_save_path, "thresh_cv.png")
                cv2.imwrite(debug_path, thresh_img)
            except Exception:
                logger.exception("Failed to write debug thresh image")

        # Convert back to PIL and return
        return _cv2_to_pil(thresh_img)

    # 4) Option B: Pillow + numpy fallback (no OpenCV)
    # apply median filter to reduce salt-and-pepper
    try:
        denoised = gray_pil.filter(ImageFilter.MedianFilter(size=3)) if denoise else gray_pil
    except Exception:
        denoised = gray_pil

    # optionally enhance contrast
    if enhance_contrast:
        try:
            enhancer = ImageEnhance.Contrast(denoised)
            denoised = enhancer.enhance(1.3)
        except Exception:
            pass

    if debug_save_path:
        denoised.save(os.path.join(debug_save_path, "denoised.png"))

    # adaptive-ish threshold using local mean window
    arr = np.array(denoised).astype("int32")
    h, w = arr.shape
    # pad and compute local mean with integral image for speed
    try:
        # block size adaptive to image size
        block_size = max(15, (min(h, w) // 40) | 1)  # odd
        # simple local mean via cv-style sliding window using uniform filter if available
        from scipy.ndimage import uniform_filter  # type: ignore
        local_mean = uniform_filter(arr.astype("float32"), size=block_size)
        thresh = (local_mean - 10).astype("int32")
        bin_arr = (arr > thresh).astype("uint8") * 255
    except Exception:
        # fallback simple global threshold based on mean
        thr = int(np.clip(arr.mean() - 10, 50, 200))
        bin_arr = (arr > thr).astype("uint8") * 255

    bin_img = Image.fromarray(bin_arr.astype("uint8"))

    if debug_save_path:
        bin_img.save(os.path.join(debug_save_path, "thresh_nopy.png"))

    return bin_img

"""OCR backend abstraction.

Defaults to pytesseract. Kept behind a single `extract_text()` function so
the backend (e.g. EasyOCR / PaddleOCR, which can handle stylized captions
better than Tesseract) can be swapped in one place if accuracy is
insufficient on a given video.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:  # pragma: no cover - exercised only when dependency missing
    pytesseract = None


# Cap the frame width before OCR. Source video here is 1920x1080 (or
# similar) — Tesseract's runtime scales with pixel count, and caption text
# doesn't need full HD resolution to read correctly. This is the single
# biggest speed lever available without changing the OCR engine itself.
MAX_OCR_WIDTH = 960

# Tesseract page-segmentation mode 11 ("sparse text — find as much text as
# possible, no particular order") suits a video frame much better than the
# default mode 3, which assumes a page of well-structured paragraphs.
TESSERACT_CONFIG = "--psm 11"


def preprocess_for_ocr(frame: np.ndarray) -> np.ndarray:
    """Downscale + grayscale + denoise + adaptive threshold.

    Burned-in video captions are usually light text with a dark outline
    over a busy, changing background — quite different from the clean
    scanned documents Tesseract is tuned for. This preprocessing
    consistently improves recognition on that kind of source, and the
    downscale step keeps each OCR call fast enough to scan a full video.
    """
    h, w = frame.shape[:2]
    if w > MAX_OCR_WIDTH:
        scale = MAX_OCR_WIDTH / w
        frame = cv2.resize(frame, (MAX_OCR_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return thresh


def extract_text(frame: np.ndarray) -> str:
    """Run OCR on a single BGR video frame and return the raw text."""
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract not installed. Run `pip install -r requirements.txt` "
            "and ensure the `tesseract-ocr` binary is on PATH."
        )
    processed = preprocess_for_ocr(frame)
    return pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)

"""Read printed text (signs, labels, medicine boxes) with Tesseract."""
import re

import cv2

from .common import load_image


def read_text(filename: str, language: str = "eng", folder: str = "input") -> dict:
    """Extract printed text from an image with OCR.

    Args:
        filename: Image filename.
        language: Tesseract language code such as 'eng', 'tel', 'hin' or 'eng+tel'.
        folder: Either 'input' or 'output'.
    """
    if not re.fullmatch(r"[a-z]{3}(\+[a-z]{3})*", language):
        raise ValueError("language must look like 'eng' or 'eng+tel'.")
    import pytesseract
    gray = load_image(filename, folder, grayscale=True)
    if max(gray.shape) < 1000:                      # small text benefits from upscaling
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    try:
        data = pytesseract.image_to_data(binary, lang=language,
                                         output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract is not installed: sudo apt install tesseract-ocr") from exc
    words, confs = [], []
    for text, conf in zip(data["text"], data["conf"]):
        if text.strip() and float(conf) >= 0:
            words.append(text.strip())
            confs.append(float(conf))
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return {"text": " ".join(words), "word_count": len(words),
            "mean_confidence": round(mean_conf, 1),
            "note": "OCR confidence is not accuracy; verify safety-critical text."}

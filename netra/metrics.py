"""Evaluation helpers: image quality (PSNR/SSIM), detection (IoU, precision, recall)."""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .common import IMAGE_IN, load_image


def compare_images(reference: str, processed: str, processed_folder: str = "output") -> dict:
    """Compute PSNR and SSIM between a reference image and a processed image.

    Args:
        reference: Reference filename inside input/images.
        processed: Processed filename.
        processed_folder: Either 'input' or 'output'.
    """
    ref = load_image(reference, "input", grayscale=True)
    proc = load_image(processed, processed_folder, grayscale=True)
    if ref.shape != proc.shape:
        raise ValueError("Image dimensions must match.")
    return {"PSNR_dB": round(float(peak_signal_noise_ratio(ref, proc, data_range=255)), 3),
            "SSIM": round(float(structural_similarity(ref, proc, data_range=255)), 4)}


def simulate_low_light(filename: str, gamma: float = 2.5, noise_sigma: float = 6.0,
                       seed: int = 0) -> dict:
    """Create a darkened, noisy copy of an image for controlled experiments.

    The copy is saved to input/images as dark_<filename> so it can be enhanced
    and compared with the original, which acts as the clean reference.

    Args:
        filename: Image filename inside input/images.
        gamma: Darkening strength from 1 to 5.
        noise_sigma: Gaussian noise standard deviation from 0 to 30.
        seed: Random seed for reproducibility.
    """
    if not 1 <= gamma <= 5 or not 0 <= noise_sigma <= 30:
        raise ValueError("gamma must be in [1, 5] and noise_sigma in [0, 30].")
    img = load_image(filename).astype(np.float32) / 255.0
    dark = np.power(img, gamma) * 255.0
    rng = np.random.default_rng(seed)
    dark += rng.normal(0, noise_sigma, dark.shape)
    out_name = f"dark_{filename}"
    if not cv2.imwrite(str(IMAGE_IN / out_name), np.clip(dark, 0, 255).astype(np.uint8)):
        raise IOError("Could not save the darkened image.")
    return {"output_input_folder": out_name,
            "parameters": {"gamma": gamma, "noise_sigma": noise_sigma, "seed": seed}}


def iou(a, b) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def precision_recall(predictions: list, truths: list, iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one matching of predictions to ground truth.

    Each item is {"label": str, "box": [x1, y1, x2, y2]}; predictions may carry
    a "confidence" used to order the matching.
    """
    preds = sorted(predictions, key=lambda p: -p.get("confidence", 1.0))
    used, tp = set(), 0
    for p in preds:
        best, best_iou = None, iou_threshold
        for i, g in enumerate(truths):
            if i in used or g["label"] != p["label"]:
                continue
            value = iou(p["box"], g["box"])
            if value >= best_iou:
                best, best_iou = i, value
        if best is not None:
            used.add(best)
            tp += 1
    fp, fn = len(preds) - tp, len(truths) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3)}

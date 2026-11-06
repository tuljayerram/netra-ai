"""Image-processing tools exposed to the agent.

Each function validates its arguments, performs one operation, saves a new
file under output/images and returns a small JSON-serialisable dictionary.
"""
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display needed (WSL, servers, CI)
import matplotlib.pyplot as plt  # noqa: E402

from .common import (IMAGE_EXT, IMAGE_IN, IMAGE_OUT, load_image, rel,  # noqa: E402
                     safe_path, save_image)


def _odd_kernel(k: int) -> int:
    if not 3 <= k <= 31 or k % 2 == 0:
        raise ValueError("kernel_size must be an odd integer from 3 to 31.")
    return k


def image_info(filename: str) -> dict:
    """Return width, height, channels, data type and file size of an input image.

    Args:
        filename: Image filename inside input/images.
    """
    path = safe_path(IMAGE_IN, filename, IMAGE_EXT)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Unable to decode image.")
    h, w = img.shape[:2]
    return {
        "filename": filename,
        "width": w,
        "height": h,
        "channels": 1 if img.ndim == 2 else img.shape[2],
        "data_type": str(img.dtype),
        "size_bytes": path.stat().st_size,
    }


def assess_image_quality(filename: str, folder: str = "input") -> dict:
    """Measure brightness, contrast and sharpness and list likely problems.

    Args:
        filename: Image filename.
        folder: Either 'input' or 'output'.
    """
    gray = load_image(filename, folder, grayscale=True)
    mean, std = float(gray.mean()), float(gray.std())
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    issues, advice = [], []
    if mean < 80:
        issues.append("low_light")
        advice.append("enhance_low_light with gamma between 1.5 and 2.2")
    elif mean > 180:
        issues.append("overexposed")
        advice.append("enhance_low_light with gamma between 0.5 and 0.8")
    if std < 35:
        issues.append("low_contrast")
        advice.append("enhance_low_light (CLAHE)")
    if sharpness < 100:
        issues.append("possibly_blurry")
        advice.append("sharpen_image, or recapture the frame")
    return {
        "filename": filename,
        "mean_intensity": round(mean, 2),
        "std_intensity": round(std, 2),
        "entropy_bits": round(entropy, 3),
        "dark_pixel_fraction": round(float((gray < 50).mean()), 3),
        "bright_pixel_fraction": round(float((gray > 205).mean()), 3),
        "laplacian_variance": round(sharpness, 2),
        "issues": issues,
        "suggested_tools": advice,
        "note": "Thresholds are heuristics; tune them on your own camera data.",
    }


def convert_grayscale(filename: str) -> dict:
    """Convert an input image to grayscale.

    Args:
        filename: Image filename inside input/images.
    """
    gray = cv2.cvtColor(load_image(filename), cv2.COLOR_BGR2GRAY)
    return {"output": rel(save_image(gray, f"gray_{filename}"))}


def enhance_low_light(filename: str, clip_limit: float = 2.0,
                      tile_grid: int = 8, gamma: float = 1.0) -> dict:
    """Improve contrast with CLAHE on the lightness channel, then apply gamma.

    Args:
        filename: Image filename inside input/images.
        clip_limit: CLAHE clip limit, 0.1 to 10.
        tile_grid: CLAHE tiles per side, 2 to 32.
        gamma: Values above 1 brighten, below 1 darken; range 0.3 to 3.
    """
    if not 0.1 <= clip_limit <= 10:
        raise ValueError("clip_limit must be in [0.1, 10].")
    if not 2 <= tile_grid <= 32:
        raise ValueError("tile_grid must be in [2, 32].")
    if not 0.3 <= gamma <= 3:
        raise ValueError("gamma must be in [0.3, 3].")
    img = load_image(filename)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_ch = clahe.apply(l_ch)
    if gamma != 1.0:
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                         dtype=np.uint8)
        l_ch = cv2.LUT(l_ch, table)
    out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    return {"output": rel(save_image(out, f"enhanced_{filename}")),
            "parameters": {"clip_limit": clip_limit, "tile_grid": tile_grid, "gamma": gamma}}


def histogram_equalization(filename: str) -> dict:
    """Apply global histogram equalization to the grayscale version of an image.

    Args:
        filename: Image filename inside input/images.
    """
    gray = load_image(filename, grayscale=True)
    return {"output": rel(save_image(cv2.equalizeHist(gray), f"equalized_{filename}"))}


def denoise_image(filename: str, method: str = "median", kernel_size: int = 5) -> dict:
    """Remove noise with a median, gaussian or bilateral filter.

    Args:
        filename: Image filename inside input/images.
        method: One of 'median', 'gaussian', 'bilateral'.
        kernel_size: Odd neighbourhood size from 3 to 31.
    """
    k = _odd_kernel(kernel_size)
    img = load_image(filename)
    if method == "median":
        out = cv2.medianBlur(img, k)
    elif method == "gaussian":
        out = cv2.GaussianBlur(img, (k, k), 0)
    elif method == "bilateral":
        out = cv2.bilateralFilter(img, k, 75, 75)
    else:
        raise ValueError("method must be 'median', 'gaussian' or 'bilateral'.")
    return {"output": rel(save_image(out, f"{method}_{filename}")),
            "parameters": {"method": method, "kernel_size": k}}


def sharpen_image(filename: str, amount: float = 1.0) -> dict:
    """Sharpen an image with an unsharp mask.

    Args:
        filename: Image filename inside input/images.
        amount: Sharpening strength from 0.1 to 3.
    """
    if not 0.1 <= amount <= 3:
        raise ValueError("amount must be in [0.1, 3].")
    img = load_image(filename)
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    out = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    return {"output": rel(save_image(out, f"sharp_{filename}")),
            "parameters": {"amount": amount}}


def canny_edges(filename: str, low_threshold: int = 100, high_threshold: int = 200) -> dict:
    """Detect edges with the Canny algorithm.

    Args:
        filename: Image filename inside input/images.
        low_threshold: Lower hysteresis threshold, 0 to 254.
        high_threshold: Upper hysteresis threshold, above low_threshold, up to 255.
    """
    if not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError("Need 0 <= low_threshold < high_threshold <= 255.")
    gray = load_image(filename, grayscale=True)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return {"output": rel(save_image(edges, f"edges_{filename}")),
            "parameters": {"low": low_threshold, "high": high_threshold}}


def threshold_segmentation(filename: str, use_otsu: bool = True,
                           threshold_value: int = 127) -> dict:
    """Segment foreground from background with a global threshold.

    Args:
        filename: Image filename inside input/images.
        use_otsu: If true, Otsu's method chooses the threshold automatically.
        threshold_value: Manual threshold from 0 to 255 when use_otsu is false.
    """
    if not 0 <= threshold_value <= 255:
        raise ValueError("threshold_value must be in [0, 255].")
    gray = load_image(filename, grayscale=True)
    flag = cv2.THRESH_BINARY | (cv2.THRESH_OTSU if use_otsu else 0)
    used, mask = cv2.threshold(gray, threshold_value, 255, flag)
    return {"output": rel(save_image(mask, f"segmented_{filename}")),
            "parameters": {"threshold_used": float(used), "otsu": use_otsu},
            "foreground_pixels": int(np.count_nonzero(mask))}


def generate_histogram(filename: str, folder: str = "input") -> dict:
    """Plot the grayscale intensity histogram of an image.

    Args:
        filename: Image filename.
        folder: Either 'input' or 'output'.
    """
    gray = load_image(filename, folder, grayscale=True)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(hist, color="#1f4e79")
    ax.set(title=f"Histogram: {filename}", xlabel="Intensity", ylabel="Pixels", xlim=(0, 255))
    out = IMAGE_OUT / f"histogram_{filename}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return {"output": rel(out)}

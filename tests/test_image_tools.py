import cv2
import numpy as np
import pytest

from netra import image_tools as it
from netra import metrics
from netra.common import BASE, IMAGE_IN


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(1)
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = (90, 120, 150)
    cv2.circle(img, (160, 120), 60, (30, 200, 240), -1)
    cv2.rectangle(img, (20, 20), (120, 90), (200, 60, 60), -1)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(IMAGE_IN / "scene.png"), img)
    return "scene.png"


def test_info(sample):
    info = it.image_info(sample)
    assert (info["width"], info["height"], info["channels"]) == (320, 240, 3)


def test_rejects_path_traversal():
    with pytest.raises(ValueError):
        it.image_info("../secret.png")
    with pytest.raises(ValueError):
        it.image_info("notes.txt")


def test_quality_flags_dark_image(sample):
    metrics.simulate_low_light(sample, gamma=3.0, noise_sigma=3)
    q = it.assess_image_quality(f"dark_{sample}")
    assert "low_light" in q["issues"]
    assert it.assess_image_quality(sample)["mean_intensity"] > q["mean_intensity"]


def test_enhancement_improves_psnr(sample):
    metrics.simulate_low_light(sample, gamma=2.5, noise_sigma=4)
    dark = f"dark_{sample}"
    out = it.enhance_low_light(dark, gamma=1.8)["output"]
    before = metrics.compare_images(sample, dark, processed_folder="input")
    after = metrics.compare_images(sample, out.split("/")[-1])
    assert after["PSNR_dB"] > before["PSNR_dB"]
    assert (BASE / out).is_file()


@pytest.mark.parametrize("method", ["median", "gaussian", "bilateral"])
def test_denoise_methods(sample, method):
    assert (BASE / it.denoise_image(sample, method, 5)["output"]).is_file()


def test_invalid_arguments(sample):
    with pytest.raises(ValueError):
        it.denoise_image(sample, "median", 4)
    with pytest.raises(ValueError):
        it.enhance_low_light(sample, clip_limit=99)
    with pytest.raises(ValueError):
        it.canny_edges(sample, 200, 100)


def test_segmentation_and_histogram(sample):
    assert it.threshold_segmentation(sample)["foreground_pixels"] > 0
    assert (BASE / it.generate_histogram(sample)["output"]).is_file()

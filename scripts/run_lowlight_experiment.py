"""Controlled low-light enhancement experiment (no network, no model needed).

Uses the astronaut image bundled with scikit-image as a clean reference,
darkens it, enhances it three ways and reports PSNR/SSIM against the original.
    python scripts/run_lowlight_experiment.py
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from skimage import data  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netra.common import BASE, IMAGE_IN, load_image  # noqa: E402
from netra.image_tools import assess_image_quality, enhance_low_light  # noqa: E402
from netra.metrics import compare_images, simulate_low_light  # noqa: E402

REF = "astronaut.png"
cv2.imwrite(str(IMAGE_IN / REF), cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
simulate_low_light(REF, gamma=2.5, noise_sigma=6, seed=0)
dark = f"dark_{REF}"

enhance_low_light(dark, 2.0, 8, 1.0)
clahe_img = load_image(f"enhanced_{dark}", "output")
cv2.imwrite(str(BASE / "output/images" / "enh_clahe.png"), clahe_img)
enhance_low_light(dark, 2.0, 8, 1.8)
gamma_img = load_image(f"enhanced_{dark}", "output")
cv2.imwrite(str(BASE / "output/images" / "enh_gamma.png"), gamma_img)
enhance_low_light(dark, 2.0, 8, 2.2)
g22 = load_image(f"enhanced_{dark}", "output")
cv2.imwrite(str(BASE / "output/images" / "enh_gamma22.png"), g22)

rows = [("Dark input", dark, "input"), ("CLAHE only (gamma 1.0)", "enh_clahe.png", "output"),
        ("CLAHE + gamma 1.8", "enh_gamma.png", "output"), ("CLAHE + gamma 2.2", "enh_gamma22.png", "output")]
results = {}
fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), gridspec_kw={"height_ratios": [3, 1.4]})
ref_img = load_image(REF)
panels = [("Original (reference)", ref_img, None)]
for title, name, folder in rows:
    img = load_image(name, folder)
    m = compare_images(REF, name, processed_folder=folder)
    results[title] = {**m, "mean_intensity": assess_image_quality(name, folder)["mean_intensity"]}
    panels.append((f"{title}\nPSNR {m['PSNR_dB']:.1f} dB, SSIM {m['SSIM']:.3f}", img, m))
for ax_img, ax_hist, (title, img, _) in zip(axes[0], axes[1], panels):
    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax_img.set_title(title, fontsize=9); ax_img.axis("off")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax_hist.hist(g.ravel(), bins=64, range=(0, 255), color="#1f4e79")
    ax_hist.set_xlim(0, 255); ax_hist.set_yticks([]); ax_hist.tick_params(labelsize=7)
plt.tight_layout()
out = BASE / "docs" / "figures" / "lowlight_experiment.png"
fig.savefig(out, dpi=170)
(BASE / "docs" / "figures" / "lowlight_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

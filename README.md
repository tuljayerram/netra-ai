# NetraAI

An image and video processing **agent for assistive vision**: a local LLM (via Ollama)
chooses from an allow-list of OpenCV / YOLO / OCR / speech tools to help analyse
camera frames for people with visual impairment. Everything runs locally on
Ubuntu (native or WSL2) with free and open-source software.

> Research and teaching prototype. It is **not** a certified mobility or safety device
> and must never replace a white cane, guide dog or orientation-and-mobility training.

*Netra* means "eye" in Sanskrit and several Indian languages.

## What it does

| Area | Tools |
|---|---|
| Image quality | `assess_image_quality`, `enhance_low_light` (CLAHE + gamma), `denoise_image`, `sharpen_image`, `histogram_equalization`, `generate_histogram` |
| Analysis | `canny_edges`, `threshold_segmentation`, `compare_images` (PSNR/SSIM), `simulate_low_light` |
| Video | `video_info`, `extract_frames`, `detect_motion` (MOG2) |
| Assistive | `detect_objects` / `guide_video` (YOLO + left/center/right + proximity), `read_text` (Tesseract OCR), `speak_text` (espeak-ng), `describe_scene` (vision-language model), `navigation_assist` (full pipeline) |

## Quick start

```bash
git clone https://github.com/<tuljayerram>/netra-ai.git && cd netra-ai
bash scripts/setup_ubuntu.sh            # apt packages + .venv + core requirements
source .venv/bin/activate
python -m pytest -q                     # 20 tests, no GPU / model needed
python scripts/run_lowlight_experiment.py

# optional: detection, local LLM
pip install -r requirements-detect.txt
curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3.1:8b
python -m netra.agent                   # chat with the agent
streamlit run app.py                    # browser interface
python -m netra.live --source street.mp4   # video file from input/videos
```

Put your own images in `input/images/` and videos in `input/videos/`
(both folders are git-ignored; do not commit photos of people).

## Layout

```
netra/        tools, spatial logic, agent, live loop
tests/        pytest suite (detection is tested with a stand-in model)
scripts/      setup script and the low-light experiment
docs/         LaTeX tutorial (netra_tutorial.tex / .pdf)
```

## Validation status

Tested automatically: image tools, video tools, spatial guidance, metrics, tool
schemas, allow-list behaviour, and detection post-processing with a fake model.
**Not exercised in the author's build environment:** real YOLO inference, Ollama
tool calling with a live model, Tesseract, espeak-ng, webcam capture and Raspberry Pi
deployment. Verify these on your machine and record your results.

## Licences

Project code: MIT (see `LICENSE`; replace the name placeholder).
Dependencies and model weights have their own licences. In particular, check the
current Ultralytics licence (AGPL-3.0 or commercial) before distributing.

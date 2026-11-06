"""Object detection with a small YOLO model, plus spatial guidance.

Ultralytics (and PyTorch) are imported lazily so the rest of the project
works without them. Check the Ultralytics licence before distributing.
"""
import json
import time

import cv2

from .common import BASE, REPORT_OUT, VIDEO_EXT, VIDEO_IN, load_image, rel, safe_path, save_image
from .spatial import annotate_detection, build_guidance

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        local = BASE / "models" / "yolo11n.pt"
        _model = YOLO(str(local) if local.is_file() else "yolo11n.pt")
    return _model


def detect_frame(frame, conf: float = 0.35, imgsz: int = 640) -> list:
    """Run the detector on a BGR frame; return label, confidence and pixel box."""
    model = _get_model()
    result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    dets = []
    for box, c, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                         result.boxes.cls.tolist()):
        dets.append({"label": model.names[int(k)], "confidence": round(float(c), 3),
                     "box": [round(v, 1) for v in box]})
    return dets


def draw_detections(frame, dets: list):
    """Draw boxes labelled with class, zone and proximity on a copy of frame."""
    h, w = frame.shape[:2]
    canvas = frame.copy()
    for d in dets:
        a = annotate_detection(d, w, h)
        x1, y1, x2, y2 = (int(v) for v in a["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f'{a["label"]} {a["confidence"]:.2f} | {a["zone"]}, {a["proximity"]}'
        cv2.putText(canvas, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def detect_objects(filename: str, conf: float = 0.35, folder: str = "input") -> dict:
    """Detect objects in an image and describe where they are.

    Args:
        filename: Image filename.
        conf: Confidence threshold from 0.05 to 0.95.
        folder: Either 'input' or 'output'.
    """
    if not 0.05 <= conf <= 0.95:
        raise ValueError("conf must be in [0.05, 0.95].")
    frame = load_image(filename, folder)
    h, w = frame.shape[:2]
    dets = detect_frame(frame, conf)
    out = save_image(draw_detections(frame, dets), f"detected_{filename}")
    return {"objects": [annotate_detection(d, w, h) for d in dets],
            "guidance": build_guidance(dets, w, h),
            "output": rel(out), "parameters": {"conf": conf}}


def guide_video(filename: str, step_seconds: float = 0.5, conf: float = 0.35) -> dict:
    """Sample a video, generate guidance per sample and save a JSON timeline.

    Args:
        filename: Video filename inside input/videos.
        step_seconds: Time between analysed frames, 0.1 to 10.
        conf: Confidence threshold from 0.05 to 0.95.
    """
    if not 0.1 <= step_seconds <= 10:
        raise ValueError("step_seconds must be in [0.1, 10].")
    if not 0.05 <= conf <= 0.95:
        raise ValueError("conf must be in [0.05, 0.95].")
    path = safe_path(VIDEO_IN, filename, VIDEO_EXT)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot open video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * step_seconds))
    timeline, latencies, count, last = [], [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count % step == 0:
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            dets = detect_frame(frame, conf)
            latencies.append((time.perf_counter() - t0) * 1000)
            message = build_guidance(dets, w, h)
            if message != last:
                timeline.append({"time_s": round(count / fps, 2), "guidance": message})
                last = message
        count += 1
        if len(latencies) >= 2000:
            break
    cap.release()
    report = REPORT_OUT / f"guidance_{path.stem}.json"
    report.write_text(json.dumps(timeline, indent=2))
    mean_ms = sum(latencies) / len(latencies) if latencies else 0.0
    return {"analysed_frames": len(latencies), "mean_latency_ms": round(mean_ms, 1),
            "messages": len(timeline), "report": rel(report), "first_messages": timeline[:5]}


def benchmark_detector(filename: str, runs: int = 10) -> dict:
    """Measure mean inference latency and frames per second on one image.

    Args:
        filename: Image filename inside input/images.
        runs: Number of timed runs, 3 to 100.
    """
    if not 3 <= runs <= 100:
        raise ValueError("runs must be in [3, 100].")
    frame = load_image(filename)
    detect_frame(frame)  # warm-up (model load, first-run allocation)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        detect_frame(frame)
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    return {"runs": runs, "mean_ms": round(mean, 1), "fps": round(1000 / mean, 2),
            "min_ms": round(min(times), 1), "max_ms": round(max(times), 1)}

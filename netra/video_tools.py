"""Video-processing tools: metadata, frame sampling and motion detection."""
import cv2

from .common import (FRAME_OUT, MAX_VIDEO_SECONDS, VIDEO_EXT, VIDEO_IN, VIDEO_OUT,
                     rel, safe_path)


def _open(filename: str):
    path = safe_path(VIDEO_IN, filename, VIDEO_EXT)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot open video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise ValueError("Invalid frame rate in video metadata.")
    if frames / fps > MAX_VIDEO_SECONDS:
        cap.release()
        raise ValueError(f"Video longer than {MAX_VIDEO_SECONDS} s.")
    return path, cap, fps, frames


def video_info(filename: str) -> dict:
    """Return resolution, frame rate, frame count and duration of an input video.

    Args:
        filename: Video filename inside input/videos.
    """
    _, cap, fps, frames = _open(filename)
    info = {
        "filename": filename,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 3),
        "total_frames": frames,
        "duration_seconds": round(frames / fps, 2),
    }
    cap.release()
    return info


def extract_frames(filename: str, interval_seconds: float = 1.0, max_frames: int = 200) -> dict:
    """Save one frame every interval_seconds from a video.

    Args:
        filename: Video filename inside input/videos.
        interval_seconds: Sampling interval in seconds, 0.04 to 60.
        max_frames: Maximum number of frames to save, 1 to 1000.
    """
    if not 0.04 <= interval_seconds <= 60:
        raise ValueError("interval_seconds must be in [0.04, 60].")
    if not 1 <= max_frames <= 1000:
        raise ValueError("max_frames must be in [1, 1000].")
    path, cap, fps, _ = _open(filename)
    out_dir = FRAME_OUT / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, round(fps * interval_seconds))
    count = saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if count % step == 0:
            target = out_dir / f"frame_{saved:05d}.jpg"
            if not cv2.imwrite(str(target), frame):
                cap.release()
                raise IOError("Could not save a frame.")
            saved += 1
        count += 1
    cap.release()
    return {"frames_read": count, "frames_saved": saved, "output_directory": rel(out_dir),
            "parameters": {"interval_seconds": interval_seconds, "frame_step": step}}


def detect_motion(filename: str, min_area_fraction: float = 0.005) -> dict:
    """Detect moving regions with MOG2 background subtraction and draw boxes.

    Works best with a still camera; a head-mounted or walking camera moves the
    whole scene, so use this only as a baseline for static-camera clips.

    Args:
        filename: Video filename inside input/videos.
        min_area_fraction: Smallest region, as a fraction of frame area, 0.0005 to 0.5.
    """
    if not 0.0005 <= min_area_fraction <= 0.5:
        raise ValueError("min_area_fraction must be in [0.0005, 0.5].")
    path, cap, fps, _ = _open(filename)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = VIDEO_OUT / f"motion_{path.stem}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create the output video (codec unavailable).")
    subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25,
                                                    detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    min_area = min_area_fraction * w * h
    total = moving = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = subtractor.apply(frame)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]   # drop shadow label (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = False
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                found = True
        moving += int(found)
        total += 1
        writer.write(frame)
    cap.release()
    writer.release()
    return {"output": rel(out), "frames_processed": total, "frames_with_motion": moving,
            "parameters": {"min_area_fraction": min_area_fraction}}

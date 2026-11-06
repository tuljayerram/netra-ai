"""Near-real-time guidance from a webcam index or a video file.

    python -m netra.live --source 0                 # webcam
    python -m netra.live --source street.mp4        # file in input/videos
"""
import argparse
import shutil
import subprocess
import time

import cv2

from .common import VIDEO_EXT, VIDEO_IN, safe_path
from .detection import detect_frame
from .spatial import GuidanceThrottle, build_guidance


def run(source: str, step: float, conf: float, speak: bool, max_seconds: float) -> None:
    target = int(source) if source.isdigit() else str(safe_path(VIDEO_IN, source, VIDEO_EXT))
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        raise SystemExit("Cannot open the video source.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, round(fps * step))
    throttle, exe, proc = GuidanceThrottle(), shutil.which("espeak-ng"), None
    start, count = time.monotonic(), 0
    try:
        while time.monotonic() - start < max_seconds:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            if count % skip:
                continue
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            message = build_guidance(detect_frame(frame, conf), w, h)
            ms = (time.perf_counter() - t0) * 1000
            if throttle.should_speak(message):
                print(f"[{time.monotonic() - start:6.1f}s | {ms:5.0f} ms] {message}")
                if speak and exe and (proc is None or proc.poll() is not None):
                    proc = subprocess.Popen([exe, "--stdin"], stdin=subprocess.PIPE, text=True)
                    proc.stdin.write(message)
                    proc.stdin.close()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--step", type=float, default=0.5, help="seconds between analysed frames")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    a = ap.parse_args()
    run(a.source, a.step, a.conf, not a.no_speech, a.max_seconds)

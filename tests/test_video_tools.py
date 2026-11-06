import cv2
import numpy as np

from netra import video_tools as vt
from netra.common import BASE, VIDEO_IN


def _make_video(name="walk.mp4", frames=40, fps=10):
    w, h = 160, 120
    writer = cv2.VideoWriter(str(VIDEO_IN / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        frame = np.full((h, w, 3), 60, np.uint8)
        cv2.rectangle(frame, (5 + 3 * i, 40), (45 + 3 * i, 90), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return name


def test_video_info_and_frames():
    name = _make_video()
    info = vt.video_info(name)
    assert info["total_frames"] == 40 and info["width"] == 160
    res = vt.extract_frames(name, interval_seconds=1.0)
    assert res["frames_saved"] == 4
    assert (BASE / res["output_directory"] / "frame_00000.jpg").is_file()


def test_motion_detection_finds_moving_box():
    name = _make_video("walk2.mp4")
    res = vt.detect_motion(name, min_area_fraction=0.01)
    assert res["frames_with_motion"] > 5
    assert (BASE / res["output"]).is_file()

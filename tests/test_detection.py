"""Detection logic tested with a stand-in model, so no PyTorch is needed in CI."""
import cv2
import numpy as np

from netra import detection
from netra.common import BASE, IMAGE_IN, VIDEO_IN


class _T(list):
    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, rows):
        self.xyxy = _T([r[0] for r in rows])
        self.conf = _T([r[1] for r in rows])
        self.cls = _T([r[2] for r in rows])


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class FakeModel:
    names = {0: "person", 1: "chair"}

    def predict(self, frame, conf, imgsz, verbose):
        h, w = frame.shape[:2]
        return [_Result([([w * 0.30, h * 0.1, w * 0.70, h * 0.95], 0.9, 0),
                         ([0.0, h * 0.6, w * 0.15, h * 0.8], 0.6, 1)])]


def test_detect_objects_with_fake_model(monkeypatch):
    monkeypatch.setattr(detection, "_get_model", lambda: FakeModel())
    cv2.imwrite(str(IMAGE_IN / "street.png"), np.full((300, 400, 3), 120, np.uint8))
    res = detection.detect_objects("street.png")
    assert res["guidance"].startswith("Person ahead, very close")
    assert {o["label"] for o in res["objects"]} == {"person", "chair"}
    assert (BASE / res["output"]).is_file()


def test_guide_video_with_fake_model(monkeypatch):
    monkeypatch.setattr(detection, "_get_model", lambda: FakeModel())
    path = str(VIDEO_IN / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(30):
        writer.write(np.full((120, 160, 3), 90, np.uint8))
    writer.release()
    res = detection.guide_video("clip.mp4", step_seconds=0.5)
    assert res["analysed_frames"] == 6 and res["messages"] == 1
    assert (BASE / res["report"]).is_file()

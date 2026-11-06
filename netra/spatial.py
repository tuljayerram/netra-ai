"""Turn raw detections into short spoken guidance.

Pure Python (no OpenCV, no model) so it is fast to test and easy to port to
a microcontroller-class device later.
"""
import time

HAZARDS = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "dog",
           "stop sign", "traffic light", "fire hydrant", "bench", "chair"}
PROX_RANK = {"very close": 3, "close": 2, "far": 1}
PHRASE = {"left": "on your left", "center": "ahead", "right": "on your right"}


def horizontal_zone(x1: float, x2: float, width: float) -> str:
    """Split the frame into left / center / right thirds by box centre."""
    cx = (x1 + x2) / 2 / width
    if cx < 1 / 3:
        return "left"
    if cx > 2 / 3:
        return "right"
    return "center"


def proximity(x1, y1, x2, y2, width, height) -> str:
    """Rough closeness from the fraction of the frame the box covers.

    This is a crude monocular proxy. It depends on object size and camera
    field of view; calibrate it, or add a depth sensor, before relying on it.
    """
    ratio = ((x2 - x1) * (y2 - y1)) / (width * height)
    if ratio >= 0.30:
        return "very close"
    if ratio >= 0.08:
        return "close"
    return "far"


def annotate_detection(det: dict, width: int, height: int) -> dict:
    x1, y1, x2, y2 = det["box"]
    out = dict(det)
    out["zone"] = horizontal_zone(x1, x2, width)
    out["proximity"] = proximity(x1, y1, x2, y2, width, height)
    return out


def build_guidance(detections: list, width: int, height: int, max_items: int = 3) -> str:
    """Rank detections and produce a short sentence for text-to-speech."""
    items = [annotate_detection(d, width, height) for d in detections]
    if not items:
        return "No known objects detected. This does not mean the way is clear."
    items.sort(key=lambda d: (-PROX_RANK[d["proximity"]],
                              d["label"] not in HAZARDS,
                              d["zone"] != "center",
                              -d["confidence"]))
    parts = []
    for d in items[:max_items]:
        text = f'{d["label"]} {PHRASE[d["zone"]]}, {d["proximity"]}'
        parts.append(text[:1].upper() + text[1:])
    return ". ".join(parts) + "."


class GuidanceThrottle:
    """Avoid repeating the same message every frame.

    Speak when the message changed (and min_gap seconds have passed) or when
    repeat_after seconds have passed since the last utterance.
    """

    def __init__(self, min_gap: float = 1.0, repeat_after: float = 8.0, clock=time.monotonic):
        self.min_gap, self.repeat_after, self.clock = min_gap, repeat_after, clock
        self.last_message = None
        self.last_time = float("-inf")

    def should_speak(self, message: str) -> bool:
        now = self.clock()
        elapsed = now - self.last_time
        changed = message != self.last_message
        if (changed and elapsed >= self.min_gap) or elapsed >= self.repeat_after:
            self.last_message, self.last_time = message, now
            return True
        return False

from netra.spatial import GuidanceThrottle, build_guidance, horizontal_zone, proximity


def test_zones():
    assert horizontal_zone(0, 100, 900) == "left"
    assert horizontal_zone(400, 500, 900) == "center"
    assert horizontal_zone(800, 890, 900) == "right"


def test_proximity_levels():
    assert proximity(0, 0, 600, 600, 1000, 1000) == "very close"   # 36% of the frame
    assert proximity(0, 0, 300, 300, 1000, 1000) == "close"        # 9%
    assert proximity(0, 0, 100, 100, 1000, 1000) == "far"


def test_guidance_prefers_closer_object():
    dets = [
        {"label": "chair", "confidence": 0.9, "box": [0, 0, 100, 100]},
        {"label": "person", "confidence": 0.8, "box": [300, 100, 700, 900]},
    ]
    text = build_guidance(dets, 1000, 1000)
    assert text.startswith("Person ahead, very close")


def test_empty_guidance_never_claims_clear_path():
    text = build_guidance([], 640, 480)
    assert "does not mean the way is clear" in text


def test_throttle():
    t = [0.0]
    th = GuidanceThrottle(min_gap=1.0, repeat_after=8.0, clock=lambda: t[0])
    assert th.should_speak("A")
    t[0] = 0.5
    assert not th.should_speak("B")     # changed but too soon
    t[0] = 1.5
    assert th.should_speak("B")
    t[0] = 3.0
    assert not th.should_speak("B")     # unchanged
    t[0] = 10.0
    assert th.should_speak("B")         # repeat interval elapsed

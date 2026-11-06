"""A deterministic pipeline: assess -> enhance (if needed) -> detect -> speak."""
from .detection import detect_objects
from .image_tools import assess_image_quality, enhance_low_light
from .speech import speak_text


def navigation_assist(filename: str, speak: bool = True) -> dict:
    """Run quality check, optional enhancement, object detection and speech.

    Args:
        filename: Image filename inside input/images.
        speak: If true, also synthesise the guidance sentence to a WAV file.
    """
    steps = []
    quality = assess_image_quality(filename)
    steps.append({"step": "assess_image_quality", "issues": quality["issues"]})
    source, folder = filename, "input"
    if {"low_light", "low_contrast"} & set(quality["issues"]):
        gamma = 1.8 if "low_light" in quality["issues"] else 1.0
        enhanced = enhance_low_light(filename, gamma=gamma)
        steps.append({"step": "enhance_low_light", **enhanced})
        source, folder = enhanced["output"].split("/")[-1], "output"
    result = detect_objects(source, folder=folder)
    steps.append({"step": "detect_objects", "output": result["output"]})
    audio = None
    if speak:
        try:
            audio = speak_text(result["guidance"])["output"]
        except RuntimeError as exc:           # espeak-ng missing: keep the text result
            steps.append({"step": "speak_text", "skipped": str(exc)})
    return {"guidance": result["guidance"], "objects": result["objects"],
            "annotated": result["output"], "audio": audio, "steps": steps}

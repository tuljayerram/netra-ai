"""Tool-calling agent: a local LLM chooses approved tools; Python runs them.

Run with:  python -m netra.agent
"""
import json
import time

from .common import BASE, LLM_MODEL, REPORT_OUT
from .detection import benchmark_detector, detect_objects, guide_video
from .image_tools import (assess_image_quality, canny_edges, convert_grayscale, denoise_image,
                          enhance_low_light, generate_histogram, histogram_equalization,
                          image_info, sharpen_image, threshold_segmentation)
from .metrics import compare_images, simulate_low_light
from .ocr_tools import read_text
from .pipeline import navigation_assist
from .speech import speak_text
from .video_tools import detect_motion, extract_frames, video_info
from .vision_tools import describe_scene

MAX_ROUNDS = 6
LOG_FILE = REPORT_OUT / "agent_log.jsonl"

TOOLS = [
    image_info, assess_image_quality, convert_grayscale, enhance_low_light,
    histogram_equalization, denoise_image, sharpen_image, canny_edges,
    threshold_segmentation, generate_histogram, compare_images, simulate_low_light,
    video_info, extract_frames, detect_motion,
    detect_objects, guide_video, benchmark_detector,
    read_text, speak_text, describe_scene, navigation_assist,
]
TOOL_MAP = {t.__name__: t for t in TOOLS}

SYSTEM_PROMPT = """You are NetraAI, an image and video processing agent for an assistive
vision research project (smart glasses for people with visual impairment).
You can only act through the provided tools.

Rules:
- Never claim an operation happened unless a tool result confirms it.
- Never invent filenames, measurements or detections.
- Ask for the filename if the user did not give one.
- Report the tool name, parameters and output location you used.
- Detections are model estimates. Never say that a path or area is safe or clear.
- You are a research aid, not a certified mobility device.
- Do not run shell commands or Python code."""


def _log(entry: dict) -> None:
    entry["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _verify(result) -> None:
    """If a tool claims to have written a file, make sure it exists."""
    if isinstance(result, dict) and isinstance(result.get("output"), str):
        if not (BASE / result["output"]).is_file():
            raise IOError(f"Tool reported {result['output']} but the file does not exist.")


def execute_tool(name: str, arguments: dict) -> dict:
    """Run one allow-listed tool; never raise, so the model can read the error."""
    func = TOOL_MAP.get(name)
    if func is None:
        result = {"error": f"Tool '{name}' is not allowed."}
        status = "rejected"
    else:
        try:
            result = func(**dict(arguments))
            _verify(result)
            status = "ok"
        except Exception as exc:                       # noqa: BLE001 - report to the model
            result, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
    _log({"tool": name, "arguments": dict(arguments), "status": status, "model": LLM_MODEL})
    return result


def run_query(query: str, history: list | None = None, max_rounds: int = MAX_ROUNDS):
    """Answer one request using at most max_rounds model calls."""
    from ollama import chat
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + (history or [])
    messages.append({"role": "user", "content": query})
    for _ in range(max_rounds):
        message = chat(model=LLM_MODEL, messages=messages, tools=TOOLS).message
        messages.append(message)
        if not message.tool_calls:
            return message.content, messages[1:]
        for call in message.tool_calls:
            result = execute_tool(call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_name": call.function.name,
                             "content": json.dumps(result, default=str)})
    return "Stopped: the tool-call limit was reached.", messages[1:]


def main() -> None:
    print("NetraAI agent. Type 'exit' to quit.")
    history: list = []
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        try:
            answer, history = run_query(query, history[-24:])
            print("\nNetraAI:", answer)
        except Exception as exc:                       # noqa: BLE001
            print("\nError:", exc)


if __name__ == "__main__":
    main()

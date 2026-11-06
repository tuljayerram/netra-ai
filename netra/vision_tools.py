"""Scene description with a local vision-language model served by Ollama."""
from .common import VISION_MODEL, load_image, safe_path, IMAGE_EXT, FOLDERS

PROMPT = ("You are helping a blind person understand a photo from a head-mounted camera. "
          "Answer briefly and concretely. Say 'unclear' instead of guessing. "
          "Never state that a path is safe or clear. ")


def describe_scene(filename: str, question: str = "Describe the scene in two sentences.",
                   folder: str = "input") -> dict:
    """Ask a vision-language model about an image.

    Args:
        filename: Image filename.
        question: What to ask about the image (at most 300 characters).
        folder: Either 'input' or 'output'.
    """
    if not question.strip() or len(question) > 300:
        raise ValueError("question must contain 1 to 300 characters.")
    load_image(filename, folder)                       # validates size and decodability
    path = safe_path(FOLDERS[folder], filename, IMAGE_EXT)
    from ollama import chat
    reply = chat(model=VISION_MODEL,
                 messages=[{"role": "user", "content": PROMPT + question,
                            "images": [str(path)]}])
    return {"answer": reply.message.content, "model": VISION_MODEL,
            "note": "Generated description; not a measurement."}

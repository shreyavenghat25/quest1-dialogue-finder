"""Vision-LLM based frame verification.

The hybrid design: Tesseract OCR does the cheap, exhaustive coarse scan
across the WHOLE video (as before) — a vision-LLM is too slow/expensive
to run on every sampled frame. But Tesseract has no semantic
understanding of what it's reading (pixel-level pattern matching only),
which is exactly why the false-positive bug earlier in this project was
possible at all: a meaningless character fragment could score as a
"match" purely on string-similarity grounds.

So: use the vision-LLM only to VERIFY a small shortlist of OCR's
top-scoring candidates, not to replace OCR outright. This keeps cost and
latency bounded (a handful of API calls per video, not thousands) while
fixing OCR's false-positive weakness at the root — the LLM actually
reasons about whether the text is really there, rather than just
computing a similarity score against noisy pixel-level output.

Requires the `anthropic` package and an ANTHROPIC_API_KEY environment
variable. Both are optional — this whole module is only touched if the
user explicitly passes --verify-with-vlm.
"""
from __future__ import annotations

import base64
import json

import cv2


def _frame_to_base64_jpeg(frame) -> str:
    """Encode a BGR numpy frame (as read by OpenCV) as a base64 JPEG
    string, the format the Claude API expects for image input."""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def verify_frame_with_vlm(frame, target_text: str, client=None,
                           model: str = "claude-haiku-4-5-20251001"):
    """Ask a vision-capable LLM whether `frame` visibly contains
    `target_text`, and what text it actually reads there.

    Returns (matched: bool, extracted_text: str, confidence: float 0-100).

    `client` is injectable specifically so this is testable without a
    real API call — pass a fake object with a `.messages.create(...)`
    method in tests. Production use (client=None) lazily creates a real
    anthropic.Anthropic() client, so importing this module doesn't
    require the `anthropic` package to be installed unless this function
    is actually called.

    A small/cheap model (Haiku-class) is used deliberately: since this
    only ever runs against a short candidate shortlist (not every frame
    in the video), a large reasoning model would be unnecessary cost for
    what's fundamentally a focused visual-verification task.
    """
    if client is None:
        import anthropic  # lazy import — only needed if this path is used
        client = anthropic.Anthropic()

    image_b64 = _frame_to_base64_jpeg(frame)

    prompt = (
        f'Does this video frame contain the on-screen text: "{target_text}"? '
        "Respond with ONLY a JSON object, no other text, in exactly this form: "
        '{"matches": true or false, "extracted_text": "<exact text you see, '
        'or empty string if none>", "confidence": <integer 0-100>}'
    )

    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, IndexError, KeyError):
        return False, "", 0.0

    return (
        bool(parsed.get("matches", False)),
        str(parsed.get("extracted_text", "")),
        float(parsed.get("confidence", 0)),
    )

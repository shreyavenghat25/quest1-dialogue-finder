import json
from unittest.mock import patch

import numpy as np

import solve
from vlm_utils import verify_frame_with_vlm


class FakeResponse:
    """Stand-in for the Anthropic SDK's response object — just enough
    shape for verify_frame_with_vlm to parse."""
    def __init__(self, text):
        self.content = [_FakeContentBlock(text)]


class _FakeContentBlock:
    def __init__(self, text):
        self.text = text


class FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self._response_text)


class FakeClient:
    def __init__(self, response_text):
        self.messages = FakeMessages(response_text)


def _make_frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_verify_frame_with_vlm_parses_positive_match():
    response_json = json.dumps({
        "matches": True,
        "extracted_text": "My mind rebels at stagnation",
        "confidence": 97,
    })
    client = FakeClient(response_json)

    matched, text, confidence = verify_frame_with_vlm(
        _make_frame(), "My mind rebels at stagnation", client=client
    )

    assert matched is True
    assert text == "My mind rebels at stagnation"
    assert confidence == 97.0


def test_verify_frame_with_vlm_parses_negative_match():
    response_json = json.dumps({
        "matches": False,
        "extracted_text": "",
        "confidence": 5,
    })
    client = FakeClient(response_json)

    matched, text, confidence = verify_frame_with_vlm(
        _make_frame(), "My mind rebels at stagnation", client=client
    )

    assert matched is False


def test_verify_frame_with_vlm_handles_malformed_response_gracefully():
    client = FakeClient("I'm not sure, this is hard to read.")

    matched, text, confidence = verify_frame_with_vlm(
        _make_frame(), "My mind rebels at stagnation", client=client
    )

    assert matched is False
    assert text == ""
    assert confidence == 0.0


def test_verify_frame_with_vlm_sends_the_target_text_in_the_prompt():
    client = FakeClient(json.dumps({"matches": False, "extracted_text": "", "confidence": 0}))

    verify_frame_with_vlm(_make_frame(), "a very specific phrase", client=client)

    sent_prompt = client.messages.calls[0]["messages"][0]["content"][1]["text"]
    assert "a very specific phrase" in sent_prompt


def test_vlm_verify_candidates_checks_chronologically_not_by_ocr_score(tmp_path):
    """The core correctness property: candidates are re-sorted to
    chronological order before verification, so an earlier true match
    wins even if a later candidate had a higher OCR score."""

    class FakeReader:
        def frame_at(self, idx):
            return f"frame-{idx}"

    reader = FakeReader()
    candidates = [
        (50, 95.0, "high scoring OCR noise"),
        (10, 60.0, "lower scoring but the real one"),
        (30, 40.0, "irrelevant"),
    ]

    def fake_verify(frame, target, client=None):
        if frame == "frame-10":
            return True, "My mind rebels at stagnation", 92.0
        return False, "", 0.0

    with patch("solve.verify_frame_with_vlm", side_effect=fake_verify):
        result = solve.vlm_verify_candidates(reader, candidates, "My mind rebels at stagnation", top_k=3)

    assert result is not None
    frame_idx, confidence, text = result
    assert frame_idx == 10


def test_vlm_verify_candidates_returns_none_when_nothing_confirmed():
    class FakeReader:
        def frame_at(self, idx):
            return f"frame-{idx}"

    reader = FakeReader()
    candidates = [(50, 95.0, "x"), (10, 60.0, "y")]

    with patch("solve.verify_frame_with_vlm", return_value=(False, "", 0.0)):
        result = solve.vlm_verify_candidates(reader, candidates, "target", top_k=5)

    assert result is None


def test_vlm_verify_candidates_respects_top_k_bound():
    """Only the top_k candidates (by OCR score) should ever be checked —
    this is what bounds API cost regardless of video length."""
    class FakeReader:
        def frame_at(self, idx):
            return f"frame-{idx}"

    reader = FakeReader()
    candidates = [(i, 100.0 - i, f"text-{i}") for i in range(20)]

    with patch("solve.verify_frame_with_vlm", return_value=(False, "", 0.0)) as mock_verify:
        solve.vlm_verify_candidates(reader, candidates, "target", top_k=3)

    assert mock_verify.call_count == 3

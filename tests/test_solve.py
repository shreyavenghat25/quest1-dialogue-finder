from unittest.mock import patch

import numpy as np

import solve


class FakeReader:
    """Stand-in for VideoReader — lets us test search logic with no real
    video file, no network, and no OCR engine installed."""

    def __init__(self, frames, fps=1.0):
        self.frames = frames
        self.fps = fps
        self.frame_count = len(frames)

    def frame_at(self, idx):
        return self.frames[idx] if 0 <= idx < len(self.frames) else None

    def timestamp_for(self, idx):
        return idx / self.fps


def _make_frames(n):
    # Distinct objects so we can key a fake OCR lookup off identity/index.
    return [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(n)]


def test_refine_finds_first_matching_frame_in_window():
    frames = _make_frames(10)
    reader = FakeReader(frames)
    target = "My mind rebels at stagnation"

    fake_texts = {
        5: "unrelated caption",
        6: target,  # first real appearance
        7: target,  # still showing
    }

    def fake_extract_text(frame):
        idx = frame[0, 0, 0]  # we encoded the frame index into pixel value
        return fake_texts.get(idx, "")

    with patch("solve.extract_text", side_effect=fake_extract_text):
        result = solve.refine(reader, target, around_frame=6)

    assert result is not None
    frame_idx, score, text = result
    assert frame_idx == 6
    assert score >= solve.MATCH_THRESHOLD


def test_refine_returns_none_when_nothing_matches():
    frames = _make_frames(10)
    reader = FakeReader(frames)

    with patch("solve.extract_text", return_value="completely unrelated text"):
        result = solve.refine(reader, "My mind rebels at stagnation", around_frame=5)

    assert result is None


def test_coarse_scan_sorts_by_score_descending():
    frames = _make_frames(6)
    reader = FakeReader(frames, fps=1.0)  # 1 fps -> coarse step = 1 frame
    target = "My mind rebels at stagnation"

    fake_texts = {0: "", 1: "", 2: target, 3: "", 4: "", 5: ""}

    def fake_extract_text(frame):
        idx = frame[0, 0, 0]
        return fake_texts.get(idx, "")

    with patch("solve.extract_text", side_effect=fake_extract_text):
        results = solve.coarse_scan(reader, target)

    assert results[0][0] == 2
    assert results[0][1] >= results[1][1]

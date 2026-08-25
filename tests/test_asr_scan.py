from unittest.mock import patch

import solve


def fake_transcribe_segments(audio_path, model_size="base", word_timestamps=True):
    """Stand-in for asr_utils.transcribe_segments — yields fixed segments
    (with word-level detail) instead of running a real Whisper model.

    Deliberately uses ONE long segment containing several unrelated words
    before the target phrase, so tests can prove word-level refine finds
    the phrase's actual start time — not just the segment's start time.
    """
    duration = 10.0
    words = [
        (0.0, 0.5, " Sherlock"), (0.5, 0.9, " said"),
        (0.9, 1.1, " My"), (1.1, 1.4, " mind"), (1.4, 1.8, " rebels"),
        (1.8, 2.0, " at"), (2.0, 2.6, " stagnation"),
        (2.6, 2.9, " give"), (2.9, 3.1, " me"), (3.1, 3.6, " problems"),
    ]
    seg_text = "Sherlock said My mind rebels at stagnation give me problems"
    yield (0.0, 3.6, seg_text, duration, words)


def fake_transcribe_no_words(audio_path, model_size="base", word_timestamps=True):
    """Segment-only transcription with no word-level detail — the
    degraded-but-still-correct fallback case."""
    yield (3.0, 6.5, "My mind rebels at stagnation", 10.0, [])


class FakeReader:
    def __init__(self, fps=25.0):
        self.fps = fps


def test_asr_scan_refines_to_word_start_not_segment_start(tmp_path):
    """The core precision fix: the segment starts at 0.0s ('Sherlock
    said...'), but the target phrase itself starts at word 'My', 0.9s in.
    The reported frame must reflect the word-level start, not 0.0s."""
    reader = FakeReader(fps=25.0)
    target = "My mind rebels at stagnation"

    with patch("asr_utils.extract_audio", return_value=tmp_path / "audio.wav"), \
         patch("asr_utils.transcribe_segments", side_effect=fake_transcribe_segments):
        result = solve.asr_scan(tmp_path / "video.mp4", reader, target, tmp_path)

    assert result is not None
    frame_idx, score, text = result
    assert frame_idx == int(0.9 * 25.0)  # word "My" start, NOT segment start (0.0s)
    assert frame_idx != int(0.0 * 25.0)
    assert score >= solve.MATCH_THRESHOLD
    assert "stagnation" in text.lower()
    assert "sherlock" not in text.lower()  # refined span excludes the unrelated lead-in


def test_asr_scan_falls_back_to_segment_start_without_word_timestamps(tmp_path):
    """If word-level detail isn't available, still return a correct (if
    coarser) result using the segment start — never fail outright."""
    reader = FakeReader(fps=25.0)
    target = "My mind rebels at stagnation"

    with patch("asr_utils.extract_audio", return_value=tmp_path / "audio.wav"), \
         patch("asr_utils.transcribe_segments", side_effect=fake_transcribe_no_words):
        result = solve.asr_scan(tmp_path / "video.mp4", reader, target, tmp_path)

    assert result is not None
    frame_idx, score, text = result
    assert frame_idx == int(3.0 * 25.0)  # segment start — the coarser fallback
    assert score >= solve.MATCH_THRESHOLD


def test_asr_scan_returns_none_when_nothing_matches(tmp_path):
    reader = FakeReader(fps=25.0)

    def unrelated_segments(audio_path, model_size="base", word_timestamps=True):
        yield (0.0, 2.0, "completely unrelated dialogue here", 2.0, [])

    with patch("asr_utils.extract_audio", return_value=tmp_path / "audio.wav"), \
         patch("asr_utils.transcribe_segments", side_effect=unrelated_segments):
        result = solve.asr_scan(tmp_path / "video.mp4", reader, "My mind rebels at stagnation", tmp_path)

    assert result is None


def test_asr_scan_writes_segments_json(tmp_path):
    reader = FakeReader(fps=25.0)

    with patch("asr_utils.extract_audio", return_value=tmp_path / "audio.wav"), \
         patch("asr_utils.transcribe_segments", side_effect=fake_transcribe_segments):
        solve.asr_scan(tmp_path / "video.mp4", reader, "My mind rebels at stagnation", tmp_path)

    segments_file = tmp_path / "asr_segments.json"
    assert segments_file.exists()
    import json
    data = json.loads(segments_file.read_text())
    assert "exited_early" in data
    assert len(data["segments"]) == 1
    assert all("start" in d and "score" in d for d in data["segments"])


def test_asr_scan_early_exit_stops_after_first_confident_segment(tmp_path):
    """The single fake segment already scores 100 and crosses threshold —
    exited_early should be recorded as True."""
    reader = FakeReader(fps=25.0)

    with patch("asr_utils.extract_audio", return_value=tmp_path / "audio.wav"), \
         patch("asr_utils.transcribe_segments", side_effect=fake_transcribe_segments):
        solve.asr_scan(tmp_path / "video.mp4", reader, "My mind rebels at stagnation", tmp_path)

    import json
    data = json.loads((tmp_path / "asr_segments.json").read_text())
    assert data["exited_early"] is True


def test_find_best_word_span_excludes_unrelated_words():
    words = [
        (0.0, 0.5, " Sherlock"), (0.5, 0.9, " said"),
        (0.9, 1.1, " My"), (1.1, 1.4, " mind"), (1.4, 1.8, " rebels"),
        (1.8, 2.0, " at"), (2.0, 2.6, " stagnation"),
        (2.6, 2.9, " give"), (2.9, 3.1, " me"), (3.1, 3.6, " problems"),
    ]
    result = solve.find_best_word_span(words, "My mind rebels at stagnation")

    assert result is not None
    score, start, end, text = result
    assert start == 0.9
    assert end == 2.6
    assert "sherlock" not in text.lower()
    assert "give" not in text.lower()

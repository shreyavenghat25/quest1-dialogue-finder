"""Audio transcription-based dialogue detection.

This is the fallback path for the ambiguity documented in APPROACH.md:
"on-screen dialogue" in the problem statement could mean a visible caption
(handled by ocr_utils.py) or a character speaking while visible in frame —
which requires finding the line in the AUDIO, not the image.

Used when ocr_utils's full-video coarse scan finds no confident match —
real signal that the phrase likely isn't rendered as visible text in this
particular video.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def extract_audio(video_path: Path, audio_path: Path, sample_rate: int = 16000) -> Path:
    """Extract mono 16kHz WAV audio from a video file via ffmpeg.

    16kHz mono is the format Whisper-family models expect internally;
    extracting to this format up front avoids letting the ASR library
    resample on the fly.
    """
    if audio_path.exists():
        return audio_path
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",                      # no video stream
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def transcribe_segments(audio_path: Path, model_size: str = "base", word_timestamps: bool = True):
    """Transcribe audio, yielding (start_seconds, end_seconds, text,
    duration, words) tuples as they're produced.

    `words` is a list of (word_start, word_end, word_text) tuples for that
    segment when word_timestamps=True (the default) — this is what makes
    frame-accurate (not just segment-accurate) output possible: a segment
    can span several seconds and multiple words, so reporting the segment's
    start time as "the" timestamp is only accurate to segment granularity.
    Word-level timestamps let the caller narrow down to exactly where the
    matched phrase itself starts, within the segment.

    Uses faster-whisper (CTranslate2-backed) rather than openai-whisper —
    meaningfully faster on CPU-only machines, which matters for a
    multi-minute-long video with no GPU available.

    A larger model_size ("small", "medium") improves accuracy at the cost
    of speed; "base" is a reasonable default for clear, single-speaker
    dialogue in English.
    """
    from faster_whisper import WhisperModel  # imported lazily — heavy dependency

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), beam_size=5, word_timestamps=word_timestamps)

    for seg in segments:
        words = [(w.start, w.end, w.word) for w in (seg.words or [])] if word_timestamps else []
        yield (seg.start, seg.end, seg.text, info.duration, words)

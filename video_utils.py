"""Utilities for downloading a video and reading frames from it."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2


def download_video(url: str, out_path: Path, insecure: bool = False) -> Path:
    """Download `url` to `out_path` using yt-dlp. Returns the path.

    Skips the download if `out_path` already exists — video files are
    large (this project's test video is ~1GB) and re-downloading on every
    iteration while tuning OCR/matching parameters wastes real time.
    Delete the file manually to force a fresh download.

    yt-dlp has a built-in extractor for ok.ru as well as YouTube and most
    other common video hosts, so this works unmodified for either.

    `insecure=True` disables TLS certificate verification (yt-dlp
    --no-check-certificate). OFF by default — this is a real security
    trade-off, not a cosmetic flag. Left as an explicit opt-in for
    networks that perform TLS interception (e.g. some campus/corporate
    WiFi, which injects a self-signed certificate into the chain and
    causes CERTIFICATE_VERIFY_FAILED for otherwise-valid HTTPS sites).
    Only use it when you've confirmed that's actually the cause (e.g. the
    same URL works fine over a different network) and you trust the
    network you're bypassing verification on.
    """
    if out_path.exists():
        print(f"  (using existing {out_path}, skipping download)")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-o", str(out_path),
        url,
    ]
    if insecure:
        print("  WARNING: TLS certificate verification disabled (--insecure). "
              "Only do this on a network you trust.")
        cmd.insert(1, "--no-check-certificate")
    subprocess.run(cmd, check=True)
    return out_path


class VideoReader:
    """Thin wrapper around cv2.VideoCapture with convenience helpers."""

    def __init__(self, path: Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frame_at(self, frame_idx: int):
        """Seek to `frame_idx` and return the decoded frame, or None."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def timestamp_for(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def release(self):
        self.cap.release()

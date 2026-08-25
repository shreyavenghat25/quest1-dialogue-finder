#!/usr/bin/env python3
"""
Find the exact frame where a given dialogue first appears in a video.

Usage:
    python solve.py --url <video_url> --text "My mind rebels at stagnation" --out results/

Design: see APPROACH.md for the full rationale. In short — a cheap coarse
scan (1 frame/sec) locates the approximate region, then a frame-by-frame
refine pass around that region finds the exact first matching frame,
without ever OCR-ing the entire video frame-by-frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from video_utils import download_video, VideoReader
from ocr_utils import extract_text
from match_utils import match_score, full_match_score

COARSE_STEP_SECONDS = 1.0
MATCH_THRESHOLD = 80  # 0-100 fuzzy score; tune per video quality
REFINE_WINDOW_SECONDS = 2.0


def coarse_scan(reader: VideoReader, target: str, step_seconds: float = COARSE_STEP_SECONDS,
                 threshold: float = MATCH_THRESHOLD, early_exit: bool = True):
    """Sample the video every `step_seconds`, OCR + score each sample.

    Returns (results, exited_early) where results is a list of
    (frame_idx, score, ocr_text) sorted by score desc.

    EARLY EXIT — not just a speed optimization, a correctness fix:
    the spec asks for the FIRST frame where the dialogue appears. Since
    sampling proceeds chronologically (ascending frame index), stopping at
    the first sample that crosses `threshold` guarantees the TRUE first
    occurrence is returned. Scanning the whole video and taking the
    globally highest-scoring frame (the old behavior) could silently
    return a LATER occurrence instead, if it happened to score marginally
    higher than an earlier true match — technically wrong against the
    spec's own wording, not just slower. Early exit closes that gap and
    speeds things up at the same time.

    Prints progress as it goes — this pass can be slow on a long video,
    and silent multi-minute loops are indistinguishable from a hang
    without visible feedback.
    """
    step_frames = max(1, int(reader.fps * step_seconds))
    sample_indices = list(range(0, reader.frame_count, step_frames))
    total = len(sample_indices)
    results = []
    start_time = time.time()
    best_so_far = 0.0
    exited_early = False

    for i, frame_idx in enumerate(sample_indices, start=1):
        frame = reader.frame_at(frame_idx)
        if frame is None:
            continue
        text = extract_text(frame)
        score = match_score(text, target)
        results.append((frame_idx, score, text))
        best_so_far = max(best_so_far, score)

        if i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else float("inf")
            sys.stdout.write(
                f"\r  scanned {i}/{total} samples "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left, best score so far: {best_so_far:.0f})"
            )
            sys.stdout.flush()

        if early_exit and score >= threshold:
            elapsed = time.time() - start_time
            skipped = total - i
            print(
                f"\n  Confident match at sample {i}/{total} (score {score:.0f}) — "
                f"stopping early, skipped {skipped} remaining samples "
                f"({skipped / total * 100:.0f}% of the video not scanned)."
            )
            exited_early = True
            break

    print()  # newline after the progress line
    results.sort(key=lambda r: r[1], reverse=True)
    return results, exited_early


def refine(reader: VideoReader, target: str, around_frame: int, threshold: float = MATCH_THRESHOLD):
    """Frame-by-frame scan in a window around `around_frame`.

    Returns the FIRST frame (walking forward from the start of the window)
    whose match score crosses `threshold`, or None if none does.
    """
    window = int(reader.fps * REFINE_WINDOW_SECONDS)
    start = max(0, around_frame - window)
    end = min(reader.frame_count - 1, around_frame + window)

    for frame_idx in range(start, end + 1):
        frame = reader.frame_at(frame_idx)
        if frame is None:
            continue
        text = extract_text(frame)
        score = match_score(text, target)
        if score >= threshold:
            return (frame_idx, score, text)
    return None


def find_best_word_span(words, target: str, threshold: float = MATCH_THRESHOLD):
    """Refine pass for the ASR path: within a segment's word list, slide a
    window across the words (sized around the target phrase's word count,
    with slack for ASR word-splitting quirks) to find the tightest span
    whose text matches the target.

    This is the audio equivalent of the OCR path's frame-level refine —
    narrowing from segment-level (coarse) to word-level (fine) precision,
    so the reported timestamp is where the matched PHRASE actually starts,
    not just where its containing segment starts (which could be several
    seconds and multiple unrelated words earlier).

    Returns (score, start_seconds, end_seconds, text) for the best span
    scoring >= threshold, or None.
    """
    target_word_count = max(1, len(target.split()))
    best = None
    n = len(words)
    for start_idx in range(n):
        # +/- slack on span length: ASR word-splitting doesn't always
        # match the target's word count exactly (e.g. contractions).
        for span in range(max(1, target_word_count - 1), target_word_count + 3):
            end_idx = start_idx + span
            if end_idx > n:
                break
            window = words[start_idx:end_idx]
            window_text = "".join(w[2] for w in window)  # whisper word tokens include leading space
            score = full_match_score(window_text, target)
            if score >= threshold and (best is None or score > best[0]):
                best = (score, window[0][0], window[-1][1], window_text.strip())
    return best


def asr_scan(video_path: Path, reader: VideoReader, target: str, out_dir: Path,
             model_size: str = "base", early_exit: bool = True):
    """Fallback search path: transcribe the audio track, coarse-match at
    segment level, then refine to word-level precision within the best
    matching segment.

    Returns (frame_idx, score, text) for the best confirmed match, or None.
    Prints progress as segments stream in — a full-video transcription can
    take several minutes and, like coarse_scan, looks hung without this.

    EARLY EXIT — same correctness argument as coarse_scan(): faster-whisper
    yields segments in chronological order, so stopping transcription at
    the FIRST segment crossing MATCH_THRESHOLD guarantees the true first
    occurrence is found, rather than transcribing the entire file and
    taking whichever segment happened to score highest overall (which
    could silently be a later occurrence). This also means a confident
    match near the start of a long video can skip transcribing the rest
    of it entirely — a real, not hypothetical, speed win.
    """
    from asr_utils import extract_audio, transcribe_segments

    audio_path = out_dir / "audio.wav"
    print("  extracting audio track...")
    extract_audio(video_path, audio_path)

    print("  transcribing (faster-whisper, word-level timestamps) — this can take several minutes on CPU...")
    start_time = time.time()
    best_segment = None  # (score, seg_start, seg_text, words)
    all_segments = []
    exited_early = False

    for seg_start, seg_end, seg_text, duration, words in transcribe_segments(audio_path, model_size=model_size):
        score = match_score(seg_text, target)
        all_segments.append({"start": seg_start, "end": seg_end, "score": score, "text": seg_text.strip()})
        if best_segment is None or score > best_segment[0]:
            best_segment = (score, seg_start, seg_text, words)

        elapsed = time.time() - start_time
        pct = min(100, seg_end / duration * 100) if duration else 0
        best_score_so_far = best_segment[0] if best_segment else 0
        sys.stdout.write(
            f"\r  transcribed up to {seg_end:6.0f}s / {duration:.0f}s ({pct:4.1f}%) "
            f"({elapsed:.0f}s elapsed, best segment score so far: {best_score_so_far:.0f})"
        )
        sys.stdout.flush()

        if early_exit and score >= MATCH_THRESHOLD:
            remaining_pct = max(0, 100 - pct)
            print(
                f"\n  Confident match at {seg_end:.0f}s (score {score:.0f}) — "
                f"stopping transcription early, skipping ~{remaining_pct:.0f}% of remaining audio."
            )
            exited_early = True
            break

    print()
    (out_dir / "asr_segments.json").write_text(json.dumps(
        {"exited_early": exited_early, "segments": all_segments}, indent=2,
    ))
    print(f"  (transcript segments saved to {out_dir / 'asr_segments.json'})")

    if best_segment is None or best_segment[0] < MATCH_THRESHOLD:
        return None

    # Refine: within the matched (coarse) segment, find the precise word
    # span, so the reported timestamp is phrase-accurate, not just
    # segment-accurate.
    score, seg_start, seg_text, words = best_segment
    print("  refining within matched segment (word-level)...")
    word_result = find_best_word_span(words, target, threshold=MATCH_THRESHOLD) if words else None

    if word_result is not None:
        w_score, w_start, _w_end, w_text = word_result
        frame_idx = int(w_start * reader.fps)
        return (frame_idx, w_score, w_text)

    # No word-level timestamps available/confirmed — fall back to the
    # segment start. Reported precision degrades to segment-level in this
    # case; still correct, just coarser.
    frame_idx = int(seg_start * reader.fps)
    return (frame_idx, score, seg_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Video URL")
    parser.add_argument("--text", required=True, help="Target dialogue text")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument(
        "--coarse-step", type=float, default=COARSE_STEP_SECONDS,
        help=f"Seconds between coarse-scan samples (default {COARSE_STEP_SECONDS}). "
             "Raise this (e.g. 2.0) for a faster first pass on a long video, at the "
             "risk of skipping a very short-lived caption.",
    )
    parser.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Minimum fuzzy match score 0-100 to accept as a confirmed match (default {MATCH_THRESHOLD}).",
    )
    parser.add_argument(
        "--insecure", action="store_true",
        help="Disable TLS certificate verification for the video download. "
             "Only use on a trusted network experiencing CERTIFICATE_VERIFY_FAILED "
             "errors caused by TLS interception (e.g. some campus/corporate WiFi).",
    )
    parser.add_argument(
        "--mode", choices=["ocr", "asr", "auto"], default="auto",
        help="'ocr' = on-screen text search only. 'asr' = spoken-audio search only. "
             "'auto' (default) = try OCR first; if no confident match is found across "
             "the whole video, automatically fall back to ASR. This hedges against the "
             "genuine ambiguity in the problem statement's phrase 'on-screen dialogue' "
             "(see APPROACH.md).",
    )
    parser.add_argument(
        "--asr-model", default="base",
        help="faster-whisper model size for ASR mode (default 'base'). "
             "Larger = more accurate, slower on CPU.",
    )
    parser.add_argument(
        "--full-scan", action="store_true",
        help="Disable early exit on BOTH the OCR coarse scan and the ASR transcription — "
             "process the entire video/audio even after finding a confident match. Off by "
             "default: early exit is both faster AND more spec-correct (guarantees the "
             "FIRST occurrence is returned, since both scans proceed chronologically — see "
             "coarse_scan()'s and asr_scan()'s docstrings). Only useful if you specifically "
             "want the full ranked candidate list, e.g. to check for multiple occurrences "
             "of the phrase.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "source.mp4"

    print("[1/4] Downloading video...")
    download_video(args.url, video_path, insecure=args.insecure)

    reader = VideoReader(video_path)
    print(f"[2/4] Video loaded: {reader.frame_count} frames @ {reader.fps:.2f} fps")

    result = None
    low_confidence = False
    method = None

    if args.mode in ("ocr", "auto"):
        print(f"[3/4] Coarse scan for candidate region (step={args.coarse_step}s)...")
        candidates, exited_early = coarse_scan(
            reader, args.text, step_seconds=args.coarse_step,
            threshold=args.threshold, early_exit=not args.full_scan,
        )

        candidates_path = out_dir / "coarse_candidates.json"
        candidates_path.write_text(json.dumps(
            {
                "exited_early": exited_early,
                "note": "Partial — scan stopped at first confident match (see exited_early)"
                        if exited_early else "Full scan of the video",
                "candidates": [{"frame": c[0], "score": c[1], "text": c[2]} for c in candidates[:20]],
            },
            indent=2,
        ))
        print(f"  (top 20 coarse candidates saved to {candidates_path})")

        if candidates and candidates[0][1] >= args.threshold - 20:
            top_frame_idx = candidates[0][0]
            print(f"[4/4] Refining around frame {top_frame_idx}...")
            result = refine(reader, args.text, top_frame_idx, threshold=args.threshold)
            if result is None:
                print("Could not confirm an exact frame within the refine window.")
                result = candidates[0]
                low_confidence = True
            method = "ocr"
        else:
            print("No strong OCR candidate found across the whole video.")
            if candidates:
                print("Top candidates:")
                for c in candidates[:5]:
                    print(f"  frame {c[0]:>6}  score {c[1]:5.1f}  text={c[2]!r}")

        if result and result[1] < args.threshold:
            low_confidence = True

    if args.mode == "asr" or (args.mode == "auto" and (result is None or result[1] < args.threshold)):
        print("[ASR] Falling back to spoken-audio search...")
        asr_result = asr_scan(video_path, reader, args.text, out_dir, model_size=args.asr_model,
                               early_exit=not args.full_scan)
        if asr_result is not None:
            result = asr_result
            low_confidence = False
            method = "asr"
        elif result is None:
            print("WARNING: no confident match found via OCR or ASR.")
            reader.release()
            return

    if result is None:
        print("WARNING: no confident match found.")
        reader.release()
        return

    frame_idx, score, text = result
    timestamp = reader.timestamp_for(frame_idx)
    frame_img = reader.frame_at(frame_idx)
    img_path = out_dir / "matched_frame.png"
    cv2.imwrite(str(img_path), frame_img)

    h, m = int(timestamp // 3600), int((timestamp % 3600) // 60)
    s = timestamp % 60

    output = {
        "timestamp": f"{h:02d}:{m:02d}:{s:06.3f}",
        "frame": frame_idx,
        "text": text.strip(),
        "confidence": score,
        "low_confidence": low_confidence,
        "detected_via": method,
        "image": str(img_path),
    }
    print(json.dumps(output, indent=2))
    (out_dir / "result.json").write_text(json.dumps(output, indent=2))

    reader.release()


if __name__ == "__main__":
    main()

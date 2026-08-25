# Dialogue Frame Finder

Given a video URL and a target dialogue string, finds the first video frame
where that dialogue is visibly on-screen, and reports the timestamp, frame
number, OCR-extracted text, and a saved image of that frame.

## How it works (quick summary)

1. **Download** the video (`yt-dlp` — supports ok.ru, YouTube, and most
   common hosts unmodified).
2. **Coarse scan**: sample 1 frame/second, run OCR on each, fuzzy-match
   against the target text. This cheaply locates the approximate region
   without OCR-ing the whole video.
3. **Refine**: frame-by-frame OCR within a small window (±2s) around the
   best coarse match, to pin down the *exact first* frame where the match
   confidence crosses the threshold.
4. **Report**: save that frame as an image, write timestamp / frame number /
   extracted text / confidence to `result.json`.

Full design rationale, trade-offs, and known limitations are in
[`APPROACH.md`](APPROACH.md). Every prompt used with an AI tool while
building this is logged in [`prompts.txt`](prompts.txt).

## Project layout

```
solve.py          # CLI entry point — orchestrates the two-phase search
video_utils.py     # video download + frame reading
ocr_utils.py        # OCR backend (Tesseract) + preprocessing
match_utils.py       # fuzzy text matching
tests/               # unit tests (no video/network/OCR-engine required)
```

## Setup

```bash
sudo apt-get install tesseract-ocr   # OCR engine (system dependency)
pip install -r requirements.txt
```

## Run

```bash
python solve.py --url "https://ok.ru/video/248244667877" \
                 --text "My mind rebels at stagnation" \
                 --out results/
```

By default this runs in `--mode auto`: tries on-screen text (OCR) search
across the whole video first, and automatically falls back to spoken-audio
(ASR) search if OCR finds nothing confident. This hedges against the
genuine ambiguity in "on-screen dialogue" — see `APPROACH.md` for why, and
for the real evidence gathered against the actual target video that
justified adding the ASR path.

Other useful flags:
```bash
--mode ocr | asr | auto     # force one detector, or use the default fallback chain
--asr-model base|small|medium  # faster-whisper model size (default: base)
--coarse-step 2.0           # faster, coarser first pass on a long video
--threshold 70               # loosen/tighten the match confidence bar
--insecure                   # only if hitting CERTIFICATE_VERIFY_FAILED from
                              # network-level TLS interception (e.g. some
                              # campus WiFi) — see the warning it prints
```

Output: `results/result.json` and `results/matched_frame.png`. Also
written: `results/coarse_candidates.json` (top 20 OCR readings, even on a
no-match run) and, if ASR ran, `results/asr_segments.json` (the full
transcript with per-segment match scores).

Example `result.json`:
```json
{
  "timestamp": "00:04:12.360",
  "frame": 6309,
  "text": "My mind rebels at stagnation",
  "confidence": 96.3,
  "low_confidence": false,
  "image": "results/matched_frame.png"
}
```

## Tests

```bash
pytest tests/ -v
```

Tests cover the fuzzy-matching logic and the coarse-scan/refine search
logic in isolation (via a fake video reader + mocked OCR), so they run in
under a second with no video file, network access, or Tesseract install
required.

## Docker (optional)

```bash
docker build -t dialogue-finder .
docker run -v $(pwd)/results:/app/results dialogue-finder \
    --url "<video_url>" --text "<dialogue>" --out results/
```

## Known limitations

See "Known limitations / possible extensions" in `APPROACH.md` — most
notably: a caption on-screen for less than the 1-second coarse-scan step
could theoretically be skipped. `APPROACH.md` describes a frame-differencing
based enhancement that would close this gap.

# Approach

## Problem restated
Given a video URL and a target dialogue string, identify:
- the first frame in which that dialogue is visibly on-screen,
- its timestamp and frame number,
- the OCR-extracted text at that frame,
- a saved image of that frame,
- and do all this without a human manually scrubbing the video.

## Design

### 1. Two-phase search: coarse scan → refine
Running OCR on every single frame of a full video is expensive and mostly
wasted work — on-screen captions/dialogue typically persist for one to
several seconds. So the search is split:

- **Coarse scan** — sample 1 frame/second, OCR each sample, fuzzy-match
  against the target. This locates the *approximate* region cheaply (a
  10-minute, 25fps video is ~15,000 frames; coarse scan only touches ~600
  of them).
- **Refine** — once the approximate region is known, OCR every frame in a
  small window (±2s) around it, walking forward to find the *first* frame
  whose match score crosses the confidence threshold. This gives
  frame-accurate output without ever OCR-ing the full video frame-by-frame.

### 2. Fuzzy matching, not exact matching
OCR on burned-in video text is noisy — compression artifacts, font/outline
rendering, motion blur, and partial occlusion all cause misreads (e.g. `0`
vs `O`, `1` vs `l`, dropped letters). Exact string comparison would fail
on near-misses like `"My m1nd rebe1s at stagnat1on"`.

`rapidfuzz.fuzz.partial_ratio`, applied after normalizing case and
punctuation, gives a 0–100 similarity score that's robust to this kind of
noise. `MATCH_THRESHOLD = 80` is the cutoff for "this is a real match" —
tuned empirically (see the unit tests, which pin down that even unrelated
text can score in the 40s-50s with `partial_ratio`, so the threshold needs
real headroom above that floor, not just "close to 100").

### 3. Preprocessing before OCR
Grayscale → bilateral filter → adaptive threshold. Video frames have busy,
constantly-changing backgrounds behind the caption text, unlike the clean
scanned documents Tesseract is tuned for by default; this preprocessing
step measurably improves recognition in practice.

### 4. Handling ambiguity / uncertainty (explicit, not silent)
- If the best coarse-scan score is far below threshold, the tool reports
  the **top 5 candidates with their scores** instead of guessing — it
  surfaces uncertainty rather than confidently returning a wrong answer.
- If refine fails to confirm a frame inside its window, the tool **falls
  back to the best coarse candidate and explicitly marks the result
  `low_confidence: true`** in the JSON output, rather than hiding the
  fact that it isn't sure.

## Known limitations / possible extensions

- **Very short captions**: if a caption is on-screen for less than the
  1-second coarse-scan step, coarse scan could in theory step over it
  entirely. A more robust (heavier) alternative: detect abrupt pixel
  changes in the likely caption region via frame differencing, and only
  OCR at those transition points. That guarantees no caption is skipped
  regardless of duration, without OCR-ing every frame either. Not
  implemented here due to time — noted as the natural next iteration, and
  the kind of change the interview process said they might ask for live.
- **OCR backend is pluggable**: `ocr_utils.py` isolates the Tesseract call
  behind a single `extract_text()` function, so swapping in EasyOCR or
  PaddleOCR (which can outperform Tesseract on stylized captions) is a
  one-file change, not a rewrite.
- **Full-frame OCR vs. subtitle-band cropping**: this scans the full frame
  rather than assuming captions sit in a fixed bottom band, so it's robust
  to dialogue appearing anywhere on screen — at the cost of somewhat more
  OCR compute per frame.

## Resolving the "on-screen dialogue" ambiguity — OCR + ASR fallback

"On-screen dialogue" in the problem statement is genuinely ambiguous: it
could mean visible caption text, OR it could use the film-terminology
sense — a character speaking while visible in frame (as opposed to
off-screen/voice-over dialogue), which says nothing about text at all.

This was resolved with evidence, not assumption: `solve.py --mode ocr`
was run against the full actual target video end-to-end. The coarse scan
completed cleanly across all ~78,000 frames with the false-positive bug
fixed, and the best score found anywhere in the entire video was ~59 —
well below the 80 confidence threshold, and every top candidate was
OCR reading visual film grain/noise, not real English text. That's a
genuine negative result across the whole video, not a partial scan or a
bug artifact.

That result is real evidence toward the audio-dialogue reading, so an
**ASR (Automatic Speech Recognition) fallback path** was added:

- `asr_utils.py` extracts the audio track (ffmpeg) and transcribes it
  with timestamps using `faster-whisper` (CTranslate2-backed — meaningfully
  faster than the reference `openai-whisper` implementation on CPU-only
  machines, which matters given multi-minute audio and no GPU).
- `solve.py`'s `asr_scan()` fuzzy-matches every transcribed segment
  against the target phrase using the *same* `match_utils.match_score()`
  already used for OCR — one matching implementation, two detection
  sources, rather than duplicating matching logic per modality.
- **`--mode auto`** (the default) tries OCR first; only if OCR finds no
  confident match across the whole video does it fall back to ASR. This
  mirrors the actual investigation process: cheap/fast detector first,
  expensive fallback only when genuinely needed.
- The final `result.json` includes a `"detected_via": "ocr" | "asr"`
  field, so the answer is never ambiguous about which detector actually
  found the match.

This is the same architecture pattern discussed earlier in design (a
shared detector interface, callback-style) — implemented for real once
there was concrete evidence it was needed, not spec pattern-matching or
premature generalization.


## Why this is meant to be extensible, not just "working"
- Video I/O, OCR, and text-matching are three separate modules, each
  independently swappable and independently testable.
- Thresholds and step sizes are named constants at the top of `solve.py`,
  not magic numbers buried in logic.
- Unit tests cover the matching logic and the coarse-scan/refine search
  logic in isolation (via a fake video reader and mocked OCR calls), so
  they run in under a second with no real video, network access, or even
  a Tesseract install — and one of them (`test_match_score_low_for_unrelated_text`)
  caught a real tuning gap during development (see `prompts.txt` / commit
  history for how that threshold was actually arrived at).

## Confirmed result on the real target video

Running `python solve.py --url "https://ok.ru/video/248244667877" --text "My mind rebels at stagnation" --out results/ --mode asr` against the
actual assignment video produced:

```json
{
  "timestamp": "00:05:21.361",
  "frame": 7705,
  "text": "My mind rebels its stagnation.",
  "confidence": 92.86,
  "low_confidence": false,
  "detected_via": "asr"
}
```

This confirms the audio-dialogue reading of "on-screen dialogue" for this
specific video — the phrase is spoken, not displayed as caption text.
Full audio transcription (faster-whisper, `base` model, CPU-only) took
~213 seconds for the ~54-minute source video.

**Note on the extracted text vs. the exact target phrase:** Whisper
transcribed "at" as "its" — a real, minor ASR mishearing, not a bug.
This is direct evidence for why fuzzy matching (`rapidfuzz.partial_ratio`,
not exact string comparison) was the right design choice throughout: an
exact-match approach would have missed this true positive entirely,
while fuzzy matching correctly identified it at 92.86% confidence.

Since the evaluators may substitute a different video/dialogue at
evaluation time (per the problem statement), `--mode auto` remains the
default behavior — this confirmed ASR result does not mean the OCR path
is now dead weight; a different evaluation video could plausibly have the
dialogue rendered as visible text instead, which is exactly why both
detectors are kept, not just the one that happened to work here.

## Word-level precision fix (ASR path)

Initially, the ASR path reported the *segment's* start time as the match
timestamp. This is only accurate to segment granularity — a segment can
span several seconds and multiple unrelated words before the target
phrase actually begins, so "exact frame" wasn't literally exact.

Fixed by adding word-level timestamps (`word_timestamps=True` in
faster-whisper) and a genuine coarse-to-refine structure for the ASR path,
mirroring the OCR path's design:

- **Coarse**: match at the segment level (as before) to find the
  approximate region
- **Refine**: within that segment's word list, slide a window sized
  around the target phrase's word count to find the tightest span whose
  text matches — `find_best_word_span()` in `solve.py`. The reported
  timestamp is that span's first word's start time, not the segment start.

**A real bug was found and fixed while building this fix.** The first
implementation reused `match_score()` (`partial_ratio`) for the word-span
comparison. Unit tests caught that this let an *incomplete* window (e.g.
"...rebels at", missing "stagnation") score competitively with the
*complete* phrase — because `partial_ratio` rewards the best-aligned
substring and doesn't penalize a candidate for being incomplete relative
to the target. Fixed by adding `full_match_score()` (`fuzz.ratio`, whole-
string comparison) in `match_utils.py`, used specifically for word-span
matching where candidate and target are expected to be close in length —
as opposed to `match_score()` (`partial_ratio`), which remains correct
for OCR/segment-level matching where the target is searched for inside
much longer, noisier surrounding text. Two different matching problems,
now two deliberately different functions, not one function misapplied to
both.

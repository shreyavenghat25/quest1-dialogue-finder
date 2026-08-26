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

## Early-exit optimization — a correctness fix, not just a speedup

Both the OCR coarse scan and the ASR transcription originally processed
the entire video/audio, then selected the globally highest-scoring
candidate. This has a subtle correctness gap against the spec's own
wording: the task asks for the **first** frame where the dialogue
appears, not the best-scoring one. If the phrase occurred more than once
in a video, and a later occurrence happened to score marginally higher
(clearer audio, better lighting, etc.), the original code would silently
return the wrong occurrence.

Since both scans proceed strictly chronologically (coarse_scan samples
frames in ascending order; faster-whisper yields transcript segments in
ascending time order), stopping at the **first** sample/segment that
crosses `MATCH_THRESHOLD` guarantees the true first occurrence is
returned — the early exit is simultaneously a speed optimization and a
correctness fix, not a trade-off between the two.

Controlled by `--full-scan` (off by default) if the complete ranked
candidate list is specifically wanted, e.g. to check whether a phrase
occurs multiple times in a video.

## Hybrid vision-LLM verification (implemented, opt-in)

Tesseract has no semantic understanding of what it's reading — it's
pixel-level pattern matching, which is exactly why the earlier
false-positive bug was possible at all (a meaningless character fragment
could score as a "match" purely on string similarity). A vision-language
model can be asked directly whether a frame contains specific text,
reasoning about the image rather than just computing a similarity score
against noisy OCR output.

Running a vision-LLM on every coarse-scan sample would be too slow and
too expensive to be practical across a whole video, so this is a hybrid,
tiered design, implemented in `vlm_utils.py` and wired into `solve.py`
behind `--verify-with-vlm`:

1. Coarse scan runs exactly as before — cheap, fast, local Tesseract,
   across the whole video. Unmodified.
2. Only the **top-K** coarse candidates by OCR score are passed to the
   vision-LLM (`--vlm-top-k`, default 5) — a small, cost-bounded set
   regardless of video length.
3. Those candidates are **re-sorted to chronological order** before
   verification (`vlm_verify_candidates()`), and checked in that order,
   stopping at the first one the LLM confirms. This is deliberate and
   tested (`test_vlm_verify_candidates_checks_chronologically_not_by_ocr_score`):
   verifying in OCR-score order could return a later frame that happened
   to have a marginally higher OCR score, which would violate the same
   "first occurrence" requirement the early-exit optimization exists to
   protect elsewhere in this project. Whichever detector produces the
   final answer, "first occurrence" should mean the same thing.
4. Uses a small/cheap model (Haiku-class) rather than a large one — this
   is a focused visual-verification task on a handful of images, not
   open-ended reasoning, so a large model would be unnecessary cost.

This mirrors a standard retrieval-then-rerank pattern — cheap search over
everything, expensive verification only on a short list — and fixes the
false-positive problem at its root (semantic understanding) rather than
only patching around it with heuristics like the length guard (which
still exists and still matters for the plain-OCR path when
`--verify-with-vlm` isn't used).

**Why this is opt-in, not the default:** it requires the `anthropic`
package, an `ANTHROPIC_API_KEY`, network access, and costs real money per
call — none of which should be silently required just to run the base
solution. `--mode auto` (OCR → ASR fallback) remains fully self-contained
and free to run; `--verify-with-vlm` is an additional layer on top for
when higher trust in the OCR path specifically is worth the cost.

**What's tested vs. what isn't:** `vlm_utils.py`'s parsing logic and
`vlm_verify_candidates()`'s chronological-ordering and top-K-bounding
behavior are covered by unit tests using a fake client (no real API
calls). What's NOT verified is a live run against the actual video with a
real Anthropic API key — that would need `ANTHROPIC_API_KEY` set and
willingness to spend API credits, which wasn't done here given time.

## Performance optimizations — sequential I/O + parallel OCR

Two additional optimizations, both correctness-preserving (verified by
tests, not just assumed):

**1. Sequential frame reading instead of per-sample seeking.**
The original coarse_scan and refine both called `frame_at(idx)` — which
internally does `cv2.CAP_PROP_POS_FRAMES` seeking — once per sample.
For compressed video (H.264/H.265), an arbitrary seek isn't O(1): the
decoder typically has to walk forward from the nearest preceding keyframe
internally, so many repeated seeks can cost far more than a single
sequential pass. `VideoReader.iter_frames()` seeks ONCE to the starting
position, then uses `cap.grab()` (cheap — advances one frame without
decoding) for frames being skipped, and `cap.retrieve()` (decodes) only
for frames actually wanted. Both `coarse_scan` and `refine` now use this.

**2. Batched, parallel OCR during the coarse scan.**
Frames are gathered into chronological batches (`--ocr-batch-size`,
default 8), and OCR runs concurrently across a batch via a thread pool
(`--ocr-workers`, default `min(8, cpu_count)`). This is genuine
parallelism, not fake: `pytesseract` invokes Tesseract as a subprocess,
which releases Python's GIL while waiting on it, so multiple OCR calls
really do run concurrently on a multi-core machine.

**Preserving the early-exit correctness guarantee under parallelism.**
`executor.map()` (not `submit()` + `as_completed()`) keeps results in
the same order as the input batch, so scanning a completed batch for the
first frame crossing the threshold still happens in chronological order
within that batch — and since batches themselves are processed strictly
in sequence, the "first occurrence" guarantee from the early-exit fix is
fully preserved, not just approximately true. This is specifically
covered by `test_coarse_scan_batched_parallel_ocr_still_finds_earliest_match`,
which sets up a later match that would "win" under naive out-of-order
parallelism and confirms the earlier one is still returned.

## OCR-positive path — verified with a synthetic test video

Every real-data confirmation up to this point came from one video: the
OCR path only ever produced a genuine NEGATIVE result on it (no on-screen
text present), and the ASR path produced the genuine POSITIVE. The
OCR-positive path — actually detecting real on-screen text — had never
been exercised against real data, only unit-tested with mocks.

Closed that gap with a small, controlled synthetic test rather than
searching for an unverified third-party video (which can't be trusted to
actually contain what it's assumed to contain without watching it
directly). Generated a 10-second local video (ffmpeg + a still frame from
Pillow) with the phrase "Hello from the test video" visible on screen
only between seconds 4–7, and nothing on screen at any other point —
a fully known, controlled ground truth.

Running `solve.py --mode ocr` against it (bypassing the network entirely,
since `download_video()` already skips re-fetching when the target file
exists locally) produced:

```json
{
  "timestamp": "00:00:04.000",
  "frame": 100,
  "text": "Helloffromitheltestivideo)",
  "confidence": 84.0,
  "low_confidence": false,
  "detected_via": "ocr"
}
```

The timestamp (00:00:04.000, frame 100 at 25fps) lands exactly at the
true start of the on-screen text window, and OCR's extracted text —
noisy but clearly the real phrase with some character-merging artifacts
— was correctly identified by fuzzy matching at 84% confidence. Early
exit also fired correctly here (stopped at coarse sample 5/10, the first
one crossing threshold, rather than scanning the full 10-second clip
unnecessarily).

This is the first real, non-mocked confirmation that the OCR path
correctly detects genuine on-screen text, not just correctly detects its
absence — closing the "designed to generalize vs. proven to generalize"
gap for the OCR-positive case specifically. Saved under
`results/ocr_positive_test/` alongside the main video's results.

"""Fuzzy matching between noisy OCR output and a target dialogue string."""
from __future__ import annotations

import re

from rapidfuzz import fuzz

# partial_ratio finds the best-matching substring of the SHORTER string's
# length inside the longer one. When the candidate is very short (e.g. a
# 1-2 character OCR fragment from a near-empty/garbage read), it can find
# a trivial perfect match purely by chance — e.g. OCR output "t\\" reduces
# to "t" after normalization, and "t" is *somewhere* inside almost any
# target phrase, so partial_ratio reports a false 100. This constant sets
# the minimum candidate length (as a fraction of the target's length)
# before a score is trusted at all.
MIN_LENGTH_RATIO = 0.6


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Makes matching robust to OCR quirks like inconsistent punctuation or
    stray characters picked up from subtitle styling.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_score(candidate: str, target: str) -> float:
    """Return a 0-100 similarity score between OCR output and the target.

    `partial_ratio` is used (rather than a plain ratio) because OCR output
    from a full video frame often contains extra noise around the actual
    caption text — we want to know if the target is *contained* in what
    was read, not whether the two strings are identical end-to-end.

    Guards against the short-candidate false-positive described above by
    returning 0 outright when the candidate is too short, relative to the
    target, to be a meaningful match.
    """
    norm_candidate = normalize(candidate)
    norm_target = normalize(target)
    if not norm_target:
        return 0.0
    if len(norm_candidate) < MIN_LENGTH_RATIO * len(norm_target):
        return 0.0
    return fuzz.partial_ratio(norm_candidate, norm_target)


def full_match_score(candidate: str, target: str) -> float:
    """Return a 0-100 whole-string similarity score (fuzz.ratio, not
    partial_ratio).

    Use this instead of match_score() when comparing two strings expected
    to be close in length end-to-end — e.g. a word-level span already
    narrowed to roughly the target's word count — rather than searching
    for the target as a substring buried in much longer noisy text.

    partial_ratio is the wrong tool here: it rewards the best-aligned
    SUBSTRING match, so a window missing a trailing word (e.g. found
    "...rebels at" but not "...at stagnation") can score equally to, or
    even above, the window that actually contains the complete phrase —
    it doesn't penalize the candidate for being incomplete relative to
    the target the way a whole-string comparison does.
    """
    norm_candidate = normalize(candidate)
    norm_target = normalize(target)
    if not norm_target:
        return 0.0
    if len(norm_candidate) < MIN_LENGTH_RATIO * len(norm_target):
        return 0.0
    return fuzz.ratio(norm_candidate, norm_target)

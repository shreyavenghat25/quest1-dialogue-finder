from match_utils import normalize, match_score, full_match_score


def test_normalize_strips_punctuation_and_case():
    assert normalize("My Mind Rebels, at STAGNATION!!") == "my mind rebels at stagnation"


def test_match_score_exact_match_is_100():
    assert match_score("My mind rebels at stagnation", "My mind rebels at stagnation") == 100


def test_match_score_tolerates_ocr_noise():
    # Typical OCR misreads: 0/O, 1/l confusion, dropped letters
    noisy = "Hy mind rebe1s at stagnat1on"
    score = match_score(noisy, "My mind rebels at stagnation")
    assert score > 70


def test_match_score_low_for_unrelated_text():
    # NOTE: partial_ratio looks for the best-matching substring, so short
    # unrelated strings can still score higher than you'd naively expect
    # (measured ~48 here, not near-0). MATCH_THRESHOLD in solve.py is set
    # to 80 specifically so this kind of noise stays well below the
    # "confirmed match" bar — this test pins that gap rather than assuming
    # unrelated text scores near zero.
    score = match_score("Subscribe and hit the bell icon", "My mind rebels at stagnation")
    assert score < 60


def test_match_score_handles_surrounding_noise():
    # Target text embedded inside other OCR-picked-up junk from the frame
    noisy = "09:14 >> \"My mind rebels at stagnation\" -- next episode"
    score = match_score(noisy, "My mind rebels at stagnation")
    assert score > 85


def test_match_score_rejects_short_garbage_as_false_positive():
    # Regression test: rapidfuzz.partial_ratio can report a false 100 when
    # the candidate is a tiny OCR fragment that trivially appears somewhere
    # inside the target (e.g. a single stray letter). Caught against real
    # OCR output: extract_text() returned "t\\" for a frame with no real
    # caption, which scored 100 before this guard was added.
    score = match_score("t\\", "My mind rebels at stagnation")
    assert score == 0.0


def test_match_score_rejects_empty_candidate():
    assert match_score("", "My mind rebels at stagnation") == 0.0


def test_full_match_score_penalizes_incomplete_span():
    # Regression case: partial_ratio let an INCOMPLETE window ("...rebels
    # at", missing "stagnation") score competitively with the COMPLETE
    # phrase, because it only rewards the best-aligned substring rather
    # than penalizing missing content. full_match_score (whole-string
    # ratio) must clearly prefer the complete match.
    target = "My mind rebels at stagnation"
    complete = "My mind rebels at stagnation"
    incomplete = "Sherlock said My mind rebels at"
    assert full_match_score(complete, target) > full_match_score(incomplete, target)


def test_full_match_score_exact_match_is_100():
    assert full_match_score("My mind rebels at stagnation", "My mind rebels at stagnation") == 100

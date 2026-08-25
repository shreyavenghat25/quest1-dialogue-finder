import numpy as np

from ocr_utils import preprocess_for_ocr


def test_preprocess_returns_single_channel_same_hw():
    frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    processed = preprocess_for_ocr(frame)
    assert processed.shape == (100, 200)


def test_preprocess_output_is_binary():
    frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    processed = preprocess_for_ocr(frame)
    unique_vals = set(np.unique(processed).tolist())
    assert unique_vals.issubset({0, 255})

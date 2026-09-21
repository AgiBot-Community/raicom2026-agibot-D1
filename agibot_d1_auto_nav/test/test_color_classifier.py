import numpy as np

from agibot_d1_auto_nav.color_classifier import classify_mask_color


def classify_uniform_hsv(hue, saturation, value):
    hsv = np.empty((10, 10, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = value
    mask = np.full((10, 10), 255, dtype=np.uint8)
    return classify_mask_color(hsv, mask)


def test_dark_olive_character_is_green():
    prediction = classify_uniform_hsv(28, 110, 90)

    assert prediction.name == 'green'


def test_bright_saturated_character_remains_yellow():
    prediction = classify_uniform_hsv(27, 170, 220)

    assert prediction.name == 'yellow'


def test_golden_yellow_character_is_not_orange():
    prediction = classify_uniform_hsv(18, 195, 175)

    assert prediction.name == 'yellow'


def test_low_hue_character_remains_orange():
    prediction = classify_uniform_hsv(12, 195, 175)

    assert prediction.name == 'orange'


def test_dark_red_orange_character_is_orange():
    prediction = classify_uniform_hsv(6, 200, 135)

    assert prediction.name == 'orange'


def test_zero_hue_character_remains_red():
    prediction = classify_uniform_hsv(1, 190, 130)

    assert prediction.name == 'red'

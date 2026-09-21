import cv2
import numpy as np
from pathlib import Path

from agibot_d1_auto_nav.traffic_light_utils import recognize_traffic_light_frame
from agibot_d1_auto_nav.traffic_light_utils import recognize_traffic_light_frames


def traffic_light_frame(color, value=255):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    if color == 'red':
        bgr = (0, 0, value)
    else:
        bgr = (0, value, 0)
    cv2.circle(frame, (160, 100), 18, bgr, -1)
    return frame


def hsv_color(hue, saturation, value):
    pixel = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    return tuple(int(channel) for channel in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def test_bright_green_light_is_recognized():
    assert recognize_traffic_light_frame(traffic_light_frame('green')).name == 'green'


def test_bright_red_light_is_recognized():
    assert recognize_traffic_light_frame(traffic_light_frame('red')).name == 'red'


def test_dim_printed_color_is_ignored():
    assert recognize_traffic_light_frame(traffic_light_frame('red', value=120)).name == 'unknown'


def test_large_dim_red_panel_beats_small_bright_green_background():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (45, 80), (275, 225), (0, 0, 135), -1)
    cv2.circle(frame, (30, 25), 8, (0, 255, 0), -1)

    assert recognize_traffic_light_frame(frame).name == 'red'


def test_dark_green_panel_beats_skin_colored_background():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 110), (115, 239), hsv_color(14, 95, 150), -1)
    cv2.rectangle(frame, (125, 55), (240, 225), hsv_color(60, 32, 61), -1)

    prediction = recognize_traffic_light_frame(frame)

    assert prediction.name == 'green'


def test_more_green_votes_are_not_overridden_by_minimum_red_votes(tmp_path):
    red_frames = [traffic_light_frame('red') for _ in range(3)]
    green_frames = [traffic_light_frame('green') for _ in range(5)]

    result = recognize_traffic_light_frames(
        red_frames + green_frames,
        output_dir=tmp_path,
        min_valid_votes=3,
    )

    assert result.state == 'green'


def test_competition_overexposed_green_lamp_beats_red_background():
    image_path = Path(__file__).resolve().parents[1] / 'tmp' / (
        'traffic_light_red_20260812_142722_674847.jpg'
    )
    frame = cv2.imread(str(image_path))

    assert frame is not None
    prediction = recognize_traffic_light_frame(frame)

    assert prediction.name == 'green'
    assert prediction.confidence >= 0.90


def test_competition_overexposed_red_lamp_stays_red():
    image_path = Path(__file__).resolve().parents[1] / 'tmp' / (
        'traffic_light_red_20260812_142121_157466.jpg'
    )
    frame = cv2.imread(str(image_path))

    assert frame is not None
    prediction = recognize_traffic_light_frame(frame)

    assert prediction.name == 'red'
    assert prediction.confidence >= 0.90

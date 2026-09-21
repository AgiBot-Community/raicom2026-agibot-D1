import numpy as np

from agibot_d1_auto_nav.qrcode_utils import decode_qrcode_frame
from agibot_d1_auto_nav.qrcode_utils import extract_target_point


def test_extract_target_point_accepts_common_payloads():
    assert extract_target_point('A') == 'A'
    assert extract_target_point('qr-a-01') == 'A'
    assert extract_target_point('TARGET: B') == 'B'
    assert extract_target_point('目标点C') == 'C'
    assert extract_target_point('destination_c_03') == 'C'


def test_extract_target_point_rejects_ambiguous_payloads():
    assert extract_target_point('') is None
    assert extract_target_point('ABC') is None
    assert extract_target_point('route A or B') is None
    assert extract_target_point('https://example.com') is None


def test_decode_qrcode_frame_reuses_injected_detector():
    class Detector:
        def __init__(self):
            self.calls = 0

        def detectAndDecode(self, _frame):
            self.calls += 1
            return 'TARGET: B', np.ones((4, 2), dtype=np.float32), None

    detector = Detector()
    contents = decode_qrcode_frame(
        np.zeros((80, 80, 3), dtype=np.uint8),
        detector=detector,
    )

    assert contents == ['TARGET: B']
    assert detector.calls > 1

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from agibot_d1_auto_nav.vision_utils import capture_frames


@dataclass(frozen=True)
class QRCodeResult:
    target: str
    content: str
    image_path: Path
    valid_frame_count: int
    total_frame_count: int


def extract_target_point(content: str) -> str | None:
    """Extract an A/B/C destination from common competition QR payloads."""
    normalized = content.strip().upper()
    if normalized in {'A', 'B', 'C'}:
        return normalized

    marked = re.search(
        r'(?:QR|TARGET|POINT|DESTINATION|目标点?|目的地)\s*[-_:：]?\s*([ABC])'
        r'(?:\b|[-_0-9])',
        normalized,
    )
    if marked:
        return marked.group(1)

    tokens = re.split(r'[^A-Z0-9]+', normalized)
    candidates = {token for token in tokens if token in {'A', 'B', 'C'}}
    if len(candidates) == 1:
        return candidates.pop()
    return None


def _image_variants(frame: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    enlarged = cv2.resize(gray, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)

    height, width = frame.shape[:2]
    margin_y = max(int(height * 0.08), 1)
    margin_x = max(int(width * 0.08), 1)
    center_crop = frame[margin_y:height - margin_y, margin_x:width - margin_x]
    return [frame, gray, clahe, otsu, adaptive, enlarged, center_crop]


def decode_qrcode_frame(
    frame: np.ndarray,
    detector: cv2.QRCodeDetector | None = None,
) -> list[str]:
    active_detector = detector if detector is not None else cv2.QRCodeDetector()
    decoded: list[str] = []
    for variant in _image_variants(frame):
        try:
            content, points, _ = active_detector.detectAndDecode(variant)
        except cv2.error:
            continue
        if points is not None and content:
            decoded.append(content.strip())
            continue

        if hasattr(active_detector, 'detectAndDecodeCurved'):
            try:
                content, points, _ = active_detector.detectAndDecodeCurved(variant)
            except cv2.error:
                continue
            if points is not None and content:
                decoded.append(content.strip())
    return list(dict.fromkeys(decoded))


def recognize_qrcode_frames(
    frames: list[np.ndarray],
    output_dir: Path,
    min_valid_votes: int = 2,
    detector: cv2.QRCodeDetector | None = None,
) -> QRCodeResult:
    if not frames:
        raise RuntimeError('no frames supplied for QR code recognition')

    votes: Counter[str] = Counter()
    contents: dict[str, Counter[str]] = {}
    representative_frames: dict[str, np.ndarray] = {}
    for frame in frames:
        frame_targets: dict[str, str] = {}
        for content in decode_qrcode_frame(frame, detector=detector):
            target = extract_target_point(content)
            if target is not None:
                frame_targets.setdefault(target, content)
        for target, content in frame_targets.items():
            votes[target] += 1
            contents.setdefault(target, Counter())[content] += 1
            representative_frames.setdefault(target, frame.copy())

    if not votes:
        raise RuntimeError(
            f'no QR code containing destination A, B, or C was decoded from {len(frames)} frames'
        )

    ranked = votes.most_common()
    target, valid_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == valid_count:
        raise RuntimeError(f'ambiguous QR destination vote: {dict(votes)}')
    if valid_count < max(int(min_valid_votes), 1):
        raise RuntimeError(
            f'QR destination {target} received only {valid_count} valid frame votes; '
            f'at least {max(int(min_valid_votes), 1)} required'
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    image_path = output_dir / f'qrcode_{target}_{timestamp}.jpg'
    cv2.imwrite(str(image_path), representative_frames[target])
    content = contents[target].most_common(1)[0][0]
    return QRCodeResult(
        target=target,
        content=content,
        image_path=image_path,
        valid_frame_count=valid_count,
        total_frame_count=len(frames),
    )


def capture_and_recognize_qrcode(
    output_dir: Path,
    rtsp_urls: list[str],
    warmup_frames: int,
    open_timeout: float,
    frame_count: int = 20,
    frame_interval_sec: float = 0.1,
    min_valid_votes: int = 2,
    detector: cv2.QRCodeDetector | None = None,
) -> QRCodeResult:
    frames, _ = capture_frames(
        rtsp_urls=rtsp_urls,
        warmup_frames=warmup_frames,
        open_timeout=open_timeout,
        frame_count=frame_count,
        frame_interval_sec=frame_interval_sec,
    )
    return recognize_qrcode_frames(
        frames=frames,
        output_dir=output_dir,
        min_valid_votes=min_valid_votes,
        detector=detector,
    )

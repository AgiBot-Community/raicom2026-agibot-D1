from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from agibot_d1_auto_nav.color_classifier import COLOR_TO_CHINESE
from agibot_d1_auto_nav.character_classifier import CharacterClassifier
from agibot_d1_auto_nav.sign_frame_adapter import prepare_sign_frame
from agibot_d1_auto_nav.sign_recognizer import SignRecognizer
from agibot_d1_auto_nav.vision_types import SignResult


DEFAULT_RTSP_URLS = [
    'rtsp://192.168.168.168:8554/test',
    'rtsp://192.168.234.1:8554/test',
]


def open_capture(rtsp_url: str, open_timeout: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)

    deadline = time.monotonic() + max(open_timeout, 0.1)
    while time.monotonic() < deadline:
        if cap.isOpened():
            return cap
        time.sleep(0.1)
    return cap


def capture_frames(
    rtsp_urls: list[str],
    warmup_frames: int,
    open_timeout: float,
    frame_count: int,
    frame_interval_sec: float,
) -> tuple[list[np.ndarray], str]:
    last_error = 'unknown camera error'
    requested_frames = max(int(frame_count), 1)

    for rtsp_url in rtsp_urls:
        cap = open_capture(rtsp_url, open_timeout)
        if not cap.isOpened():
            last_error = f'failed to open stream: {rtsp_url}'
            cap.release()
            continue

        frames: list[np.ndarray] = []
        try:
            for _ in range(max(int(warmup_frames), 0)):
                ok, _ = cap.read()
                if not ok:
                    break

            while len(frames) < requested_frames:
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    break
                frames.append(frame.copy())
                if len(frames) < requested_frames and frame_interval_sec > 0.0:
                    time.sleep(frame_interval_sec)

            if frames:
                return frames, rtsp_url
            last_error = f'opened stream but failed to read frame: {rtsp_url}'
        finally:
            cap.release()

    raise RuntimeError(last_error)


def recognize_frame(frame: np.ndarray) -> SignResult:
    """Compatibility helper for one frame; multi-frame voting is preferred."""
    recognizer = SignRecognizer(min_valid_frames=1)
    observation = recognizer.recognize_frame(frame)
    result = SignResult(
        number=observation.digit.prediction.char,
        number_color=observation.digit.candidate.color.name,
        letter=observation.letter.prediction.char,
        letter_color=observation.letter.candidate.color.name,
        image_path=Path(),
        annotated_path=Path(),
        number_confidence=observation.digit.prediction.confidence,
        number_color_confidence=observation.digit.candidate.color.confidence,
        letter_confidence=observation.letter.prediction.confidence,
        letter_color_confidence=observation.letter.candidate.color.confidence,
        valid_frame_count=1,
        total_frame_count=1,
    )
    return result


def capture_and_recognize_sign(
    output_dir: Path,
    rtsp_urls: list[str],
    warmup_frames: int,
    open_timeout: float,
    frame_count: int = 20,
    frame_interval_sec: float = 0.1,
    min_valid_frames: int = 3,
    min_char_vote_ratio: float = 0.55,
    min_color_vote_ratio: float = 0.50,
    save_all_frames: bool = False,
    preprocess_enabled: bool = False,
    preprocess_saturation_threshold: int = 100,
    digit_position: str = 'right',
    api_key: str = '',
    base_url: str = '',
    model_name: str = '',
    prompt: str = '',
    classifier: CharacterClassifier | None = None,
) -> SignResult:
    """Capture and recognize a sign while preserving the original public API."""
    del api_key, base_url, model_name, prompt
    frames, _ = capture_frames(
        rtsp_urls=rtsp_urls,
        warmup_frames=warmup_frames,
        open_timeout=open_timeout,
        frame_count=frame_count,
        frame_interval_sec=frame_interval_sec,
    )
    return recognize_sign_frames(
        frames=frames,
        output_dir=output_dir,
        min_valid_frames=min_valid_frames,
        min_char_vote_ratio=min_char_vote_ratio,
        min_color_vote_ratio=min_color_vote_ratio,
        save_all_frames=save_all_frames,
        preprocess_enabled=preprocess_enabled,
        preprocess_saturation_threshold=preprocess_saturation_threshold,
        digit_position=digit_position,
        classifier=classifier,
    )


def _frame_rejection_reason(recognizer: SignRecognizer, frame: np.ndarray) -> str:
    try:
        observation = recognizer.recognize_frame(frame)
    except Exception as exc:
        return str(exc)
    checks = (
        ('digit character', observation.digit.prediction.confidence, recognizer.min_character_confidence),
        ('letter character', observation.letter.prediction.confidence, recognizer.min_character_confidence),
        ('digit color', observation.digit.candidate.color.confidence, recognizer.min_color_confidence),
        ('letter color', observation.letter.candidate.color.confidence, recognizer.min_color_confidence),
    )
    failed = [
        f'low {name} confidence {value:.2f} < {minimum:.2f}'
        for name, value, minimum in checks
        if value < minimum
    ]
    return '; '.join(failed) if failed else 'frame accepted individually; aggregate vote failed'


def _write_failure_diagnostics(
    recognizer: SignRecognizer,
    raw_frames: list[np.ndarray],
    prepared_frames: list[np.ndarray],
    output_dir: Path,
    error: Exception,
) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    reasons = [
        _frame_rejection_reason(recognizer, frame)
        for frame in prepared_frames
    ]
    counts = Counter(reasons)
    lines = [
        f'aggregate_error: {error}',
        f'total_frames: {len(prepared_frames)}',
        'summary:',
    ]
    lines.extend(f'  {count}x {reason}' for reason, count in counts.most_common())
    lines.append('frames:')
    lines.extend(f'  frame {index}: {reason}' for index, reason in enumerate(reasons))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f'sign_failure_{timestamp}.txt'
    log_path.write_text('\n'.join(lines), encoding='utf-8')
    for index, (raw, prepared) in enumerate(
        zip(raw_frames[:3], prepared_frames[:3]),
        start=1,
    ):
        cv2.imwrite(str(output_dir / f'sign_failure_{timestamp}_{index:02d}_raw.jpg'), raw)
        cv2.imwrite(
            str(output_dir / f'sign_failure_{timestamp}_{index:02d}_prepared.jpg'),
            prepared,
        )
    return log_path


def recognize_sign_frames(
    frames: list[np.ndarray],
    output_dir: Path,
    min_valid_frames: int = 3,
    min_char_vote_ratio: float = 0.55,
    min_color_vote_ratio: float = 0.50,
    save_all_frames: bool = False,
    preprocess_enabled: bool = False,
    preprocess_saturation_threshold: int = 100,
    digit_position: str = 'right',
    classifier: CharacterClassifier | None = None,
) -> SignResult:
    if not frames:
        raise RuntimeError('no frames supplied for sign recognition')
    prepared_frames = (
        [
            prepare_sign_frame(frame, preprocess_saturation_threshold)
            for frame in frames
        ]
        if preprocess_enabled
        else [frame.copy() for frame in frames]
    )
    recognizer = SignRecognizer(
        min_valid_frames=min_valid_frames,
        min_char_vote_ratio=min_char_vote_ratio,
        min_color_vote_ratio=min_color_vote_ratio,
        digit_position=digit_position,
        classifier=classifier,
    )
    try:
        return recognizer.recognize_frames(
            frames=prepared_frames,
            output_dir=Path(output_dir),
            save_all_frames=save_all_frames,
        )
    except Exception as exc:
        log_path = _write_failure_diagnostics(
            recognizer,
            frames,
            prepared_frames,
            Path(output_dir),
            exc,
        )
        raise RuntimeError(f'{exc}; diagnostics saved to {log_path}') from exc


def color_to_chinese(color_name: str) -> str:
    return COLOR_TO_CHINESE.get(color_name, color_name)

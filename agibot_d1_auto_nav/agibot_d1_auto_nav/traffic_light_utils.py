from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from agibot_d1_auto_nav.color_classifier import COLOR_BGR
from agibot_d1_auto_nav.vision_types import ColorPrediction
from agibot_d1_auto_nav.vision_utils import capture_frames


@dataclass(frozen=True)
class TrafficLightResult:
    state: str
    confidence: float
    image_path: Path
    valid_frame_count: int
    total_frame_count: int


def _recognize_overexposed_lamp(
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
) -> ColorPrediction | None:
    """Classify a white lamp core from the colored halo around its boundary."""
    frame_height, frame_width = hue.shape
    frame_area = frame_height * frame_width
    bright_core = ((value >= 245) & (saturation <= 70)).astype(np.uint8) * 255
    bright_core = cv2.morphologyEx(
        bright_core,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright_core,
        8,
    )
    best_prediction = None
    best_support = 0
    ring_kernel_size = max(21, int(round(min(frame_height, frame_width) * 0.04)))
    if ring_kernel_size % 2 == 0:
        ring_kernel_size += 1
    ring_kernel = np.ones((ring_kernel_size, ring_kernel_size), dtype=np.uint8)

    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if not frame_area * 0.002 <= area <= frame_area * 0.08:
            continue
        aspect_ratio = width / max(float(height), 1.0)
        fill_ratio = area / max(float(width * height), 1.0)
        if not 0.75 <= aspect_ratio <= 2.2 or fill_ratio < 0.45:
            continue
        center_x, center_y = centroids[label]
        if not (frame_width * 0.25 <= center_x <= frame_width * 0.75):
            continue
        if not (frame_height * 0.20 <= center_y <= frame_height * 0.85):
            continue

        core = (labels == label).astype(np.uint8)
        halo = (cv2.dilate(core, ring_kernel) > 0) & (core == 0)
        colored_halo = halo & (saturation >= 60) & (value >= 80)
        red_support = int(np.count_nonzero(
            colored_halo & ((hue <= 18) | (hue >= 170))
        ))
        green_support = int(np.count_nonzero(
            colored_halo & (hue >= 35) & (hue <= 100)
        ))
        total_support = red_support + green_support
        minimum_support = max(80, int(area * 0.015))
        if total_support < minimum_support:
            continue
        state = 'red' if red_support > green_support else 'green'
        strongest_support = max(red_support, green_support)
        dominance = strongest_support / max(float(total_support), 1.0)
        if dominance < 0.72 or strongest_support <= best_support:
            continue
        best_support = strongest_support
        confidence = float(np.clip(0.70 + 0.30 * dominance, 0.0, 1.0))
        best_prediction = ColorPrediction(
            state,
            confidence,
            dominance,
            int(area),
            COLOR_BGR[state],
        )
    return best_prediction


def _strongest_component(
    mask: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    min_area: int,
) -> tuple[float, int]:
    cleaned = cv2.morphologyEx(
        mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    frame_area = mask.shape[0] * mask.shape[1]
    best_score = 0.0
    best_area = 0
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        # Competition signals may be large colored panels close to the camera.
        # Only reject components that cover almost the entire image.
        if area < min_area or area > frame_area * 0.65:
            continue
        aspect_ratio = width / max(float(height), 1.0)
        fill_ratio = area / max(float(width * height), 1.0)
        if not 0.35 <= aspect_ratio <= 2.8 or fill_ratio < 0.20:
            continue

        component = labels == label
        mean_saturation = float(np.mean(saturation[component])) / 255.0
        mean_value = float(np.mean(value[component])) / 255.0
        shape_quality = min(fill_ratio / 0.75, 1.0)
        score = area * mean_saturation * mean_value * (0.5 + 0.5 * shape_quality)
        if score > best_score:
            best_score = score
            best_area = int(area)
    return best_score, best_area


def _strongest_panel_component(
    mask: np.ndarray,
    min_area: int,
) -> tuple[float, int]:
    """Score a large, dim color panel without favoring brighter background objects."""
    cleaned = cv2.morphologyEx(
        mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
    frame_height, frame_width = mask.shape
    frame_area = frame_height * frame_width
    best_score = 0.0
    best_area = 0
    for label in range(1, count):
        _, _, width, height, area = stats[label]
        if area < min_area or area > frame_area * 0.65:
            continue
        aspect_ratio = width / max(float(height), 1.0)
        fill_ratio = area / max(float(width * height), 1.0)
        if not 0.35 <= aspect_ratio <= 3.2 or fill_ratio < 0.30:
            continue

        center_x = float(centroids[label][0]) / max(float(frame_width), 1.0)
        center_quality = max(0.20, 1.0 - abs(center_x - 0.5) / 0.5)
        shape_quality = min(fill_ratio / 0.75, 1.0)
        score = area * (0.5 + 0.5 * shape_quality) * center_quality
        if score > best_score:
            best_score = score
            best_area = int(area)
    return best_score, best_area


def recognize_traffic_light_frame(
    frame: np.ndarray,
    saturation_threshold: int = 80,
    value_threshold: int = 145,
    min_component_area: int = 20,
) -> ColorPrediction:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    overexposed_lamp = _recognize_overexposed_lamp(hue, saturation, value)
    if overexposed_lamp is not None:
        return overexposed_lamp
    bright_green = (
        (saturation >= saturation_threshold) & (value >= value_threshold)
    )
    # The red panel is noticeably darker under the competition camera than the
    # illuminated green exit signs in the background.
    red_value_threshold = max(125, int(value_threshold) - 15)
    visible_red = (
        (saturation >= saturation_threshold) & (value >= red_value_threshold)
    )
    minimum_area = max(
        int(min_component_area),
        int(frame.shape[0] * frame.shape[1] * 0.0001),
    )
    color_masks = {
        'red': visible_red & ((hue <= 12) | (hue >= 170)),
        'green': bright_green & (hue >= 40) & (hue <= 88),
    }
    evidence = {
        name: _strongest_component(mask, saturation, value, minimum_area)
        for name, mask in color_masks.items()
    }

    # The competition target can be a large colored panel rather than a bright
    # lamp. Its green surface is dark and weakly saturated in the camera image,
    # so evaluate large components separately. Restrict this path to the central
    # field of view and score geometry instead of brightness; this prevents skin
    # and cardboard near the image edge from overpowering a dim green panel.
    frame_height, frame_width = hue.shape
    panel_roi = np.zeros_like(hue, dtype=bool)
    panel_roi[
        int(frame_height * 0.15):int(frame_height * 0.98),
        int(frame_width * 0.22):int(frame_width * 0.78),
    ] = True
    visible_panel = panel_roi & (saturation >= 20) & (value >= 40)
    panel_minimum_area = max(
        int(min_component_area),
        int(frame_height * frame_width * 0.02),
    )
    panel_masks = {
        'red': visible_panel & ((hue <= 18) | (hue >= 170)),
        'green': visible_panel & (hue >= 25) & (hue <= 100),
    }
    for name, mask in panel_masks.items():
        panel_score, panel_area = _strongest_panel_component(
            mask,
            panel_minimum_area,
        )
        # Keep large-panel scores on a scale that decisively exceeds incidental
        # bright points while preserving the normal bright-lamp detector.
        if panel_score > evidence[name][0]:
            evidence[name] = panel_score, panel_area
    ranked = sorted(evidence.items(), key=lambda item: item[1][0], reverse=True)
    best_name, (best_score, best_area) = ranked[0]
    second_score = ranked[1][1][0]
    if best_score <= 0.0:
        return ColorPrediction('unknown', 0.0, 0.0, 0, COLOR_BGR['unknown'])

    dominance = best_score / max(best_score + second_score, 1e-6)
    area_support = min(best_area / max(minimum_area * 3.0, 1.0), 1.0)
    confidence = float(np.clip(0.65 * dominance + 0.35 * area_support, 0.0, 1.0))
    if dominance < 0.58:
        return ColorPrediction(
            'unknown',
            confidence * 0.5,
            dominance,
            best_area,
            COLOR_BGR['unknown'],
        )
    return ColorPrediction(
        best_name,
        confidence,
        dominance,
        best_area,
        COLOR_BGR[best_name],
    )


def recognize_traffic_light_frames(
    frames: list[np.ndarray],
    output_dir: Path,
    min_valid_votes: int = 3,
    min_confidence: float = 0.55,
    saturation_threshold: int = 80,
    value_threshold: int = 145,
    min_component_area: int = 20,
) -> TrafficLightResult:
    if not frames:
        raise RuntimeError('no frames supplied for traffic light recognition')

    votes: Counter[str] = Counter()
    confidence_totals: Counter[str] = Counter()
    representative_frames: dict[str, np.ndarray] = {}
    for frame in frames:
        prediction = recognize_traffic_light_frame(
            frame,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
            min_component_area=min_component_area,
        )
        if prediction.name not in {'red', 'green'} or prediction.confidence < min_confidence:
            continue
        votes[prediction.name] += 1
        confidence_totals[prediction.name] += prediction.confidence
        representative_frames.setdefault(prediction.name, frame.copy())

    if not votes:
        raise RuntimeError(
            f'no valid red or green traffic light detected in {len(frames)} frames'
        )
    required_votes = max(int(min_valid_votes), 1)
    ranked = sorted(
        votes,
        key=lambda state: (votes[state], confidence_totals[state]),
        reverse=True,
    )
    state = ranked[0]
    valid_count = votes[state]
    if len(ranked) > 1 and votes[ranked[1]] == valid_count:
        raise RuntimeError(f'ambiguous traffic light vote: {dict(votes)}')
    if valid_count < required_votes:
        raise RuntimeError(
            f'traffic light {state} received only {valid_count} valid frame votes; '
            f'at least {required_votes} required'
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    image_path = output_dir / f'traffic_light_{state}_{timestamp}.jpg'
    if not cv2.imwrite(str(image_path), representative_frames[state]):
        raise RuntimeError(f'failed to save traffic light image: {image_path}')
    confidence = confidence_totals[state] / valid_count
    return TrafficLightResult(
        state=state,
        confidence=float(confidence),
        image_path=image_path,
        valid_frame_count=valid_count,
        total_frame_count=len(frames),
    )


def capture_and_recognize_traffic_light(
    output_dir: Path,
    rtsp_urls: list[str],
    warmup_frames: int,
    open_timeout: float,
    frame_count: int = 15,
    frame_interval_sec: float = 0.1,
    min_valid_votes: int = 3,
    min_confidence: float = 0.55,
    saturation_threshold: int = 80,
    value_threshold: int = 145,
    min_component_area: int = 20,
) -> TrafficLightResult:
    frames, _ = capture_frames(
        rtsp_urls=rtsp_urls,
        warmup_frames=warmup_frames,
        open_timeout=open_timeout,
        frame_count=frame_count,
        frame_interval_sec=frame_interval_sec,
    )
    return recognize_traffic_light_frames(
        frames=frames,
        output_dir=output_dir,
        min_valid_votes=min_valid_votes,
        min_confidence=min_confidence,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
        min_component_area=min_component_area,
    )

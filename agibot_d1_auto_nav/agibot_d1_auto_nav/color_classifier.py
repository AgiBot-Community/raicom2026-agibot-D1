from __future__ import annotations

import cv2
import numpy as np

from agibot_d1_auto_nav.vision_types import ColorPrediction


COLOR_BGR = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'purple': (255, 0, 255),
    'orange': (0, 165, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'unknown': (128, 128, 128),
}

COLOR_TO_CHINESE = {
    'red': '红色',
    'yellow': '黄色',
    'green': '绿色',
    'blue': '蓝色',
    'purple': '紫色',
    'orange': '橙色',
    'white': '白色',
    'black': '黑色',
    'gray': '灰色',
    'unknown': '未知颜色',
}


def build_chromatic_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return HSV image and a broad mask for saturated colored targets."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    mask = ((saturation >= 45) & (value >= 40)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    return hsv, mask


def _color_scores(
    h: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
) -> dict[str, int]:
    olive_green = (
        valid
        & (h >= 24)
        & (h <= 39)
        & (s <= 155)
        & (v <= 155)
    )
    return {
        'red': int(np.count_nonzero(valid & ((h <= 2) | (h >= 172)))),
        # Warm illumination shifts yellow cards toward H=13..19. Keep the
        # low-hue orange band for genuinely orange targets such as H=5..8.
        'orange': int(np.count_nonzero(valid & (h >= 3) & (h <= 12))),
        'yellow': int(np.count_nonzero(
            valid & (h >= 13) & (h <= 39) & ~olive_green
        )),
        'green': int(np.count_nonzero(
            (valid & (h >= 40) & (h <= 88)) | olive_green
        )),
        'blue': int(np.count_nonzero(valid & (h >= 89) & (h <= 132))),
        'purple': int(np.count_nonzero(valid & (h >= 133) & (h <= 171))),
    }


def classify_mask_color(
    hsv_roi: np.ndarray,
    object_mask: np.ndarray,
    min_pixels: int = 40,
) -> ColorPrediction:
    """Classify color only inside an object mask, never from its background."""
    h, s, v = cv2.split(hsv_roi)
    valid = (object_mask > 0) & (s >= 45) & (v >= 35)
    pixel_count = int(np.count_nonzero(valid))
    if pixel_count < min_pixels:
        return ColorPrediction('unknown', 0.0, 0.0, pixel_count, COLOR_BGR['unknown'])

    scores = _color_scores(h, s, v, valid)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    dominance = best_count / max(pixel_count, 1)
    separation = (best_count - second_count) / max(best_count, 1)
    confidence = float(np.clip(0.70 * dominance + 0.30 * separation, 0.0, 1.0))

    if best_count < min_pixels or dominance < 0.48:
        return ColorPrediction(
            'unknown',
            confidence * 0.5,
            dominance,
            pixel_count,
            COLOR_BGR['unknown'],
        )
    return ColorPrediction(
        best_name,
        confidence,
        dominance,
        pixel_count,
        COLOR_BGR[best_name],
    )


def classify_dominant_object_color(
    image: np.ndarray,
    object_mask: np.ndarray | None = None,
) -> ColorPrediction:
    """Reusable color attribute classifier for a detected table/chair ROI."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if object_mask is None:
        height, width = image.shape[:2]
        object_mask = np.zeros((height, width), dtype=np.uint8)
        x0, x1 = int(width * 0.15), int(width * 0.85)
        y0, y1 = int(height * 0.15), int(height * 0.85)
        object_mask[y0:y1, x0:x1] = 255
    return classify_mask_color(hsv, object_mask, min_pixels=80)


def classify_traffic_light_state(image: np.ndarray) -> ColorPrediction:
    """Classify a traffic-light ROI after an object detector has cropped it."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = (s >= 80) & (v >= 125)
    pixel_count = int(np.count_nonzero(bright))
    if pixel_count < max(8, image.shape[0] * image.shape[1] // 500):
        return ColorPrediction('unknown', 0.0, 0.0, pixel_count, COLOR_BGR['unknown'])

    scores = {
        'red': int(np.count_nonzero(bright & ((h <= 9) | (h >= 172)))),
        'yellow': int(np.count_nonzero(bright & (h >= 19) & (h <= 39))),
        'green': int(np.count_nonzero(bright & (h >= 40) & (h <= 88))),
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_count = ranked[0]
    second_count = ranked[1][1]
    dominance = best_count / max(pixel_count, 1)
    separation = (best_count - second_count) / max(best_count, 1)
    confidence = float(np.clip(0.65 * dominance + 0.35 * separation, 0.0, 1.0))
    if dominance < 0.42:
        return ColorPrediction('unknown', confidence * 0.5, dominance, pixel_count, COLOR_BGR['unknown'])
    return ColorPrediction(best_name, confidence, dominance, pixel_count, COLOR_BGR[best_name])

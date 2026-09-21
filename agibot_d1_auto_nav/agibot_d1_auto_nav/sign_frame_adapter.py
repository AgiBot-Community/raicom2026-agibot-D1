from __future__ import annotations

import cv2
import numpy as np


CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
DISPLAY_WIDTH_RATIO = 0.52
DISPLAY_HEIGHT_RATIO = 0.67


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def _quad_from_contour(contour: np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return _order_quad(approx.reshape(4, 2))
    return _order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))


def find_bright_display_quad(frame: np.ndarray) -> np.ndarray | None:
    """Locate the bright screen/paper without changing the recognizer itself."""
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bright_surface = cv2.inRange(hsv, (0, 0, 125), (179, 145, 255))
    scale = max(min(width, height) / 480.0, 1.0)
    close_size = max(int(round(17 * scale)) | 1, 9)
    open_size = max(int(round(5 * scale)) | 1, 3)
    bright_surface = cv2.morphologyEx(
        bright_surface,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), dtype=np.uint8),
    )
    bright_surface = cv2.morphologyEx(
        bright_surface,
        cv2.MORPH_OPEN,
        np.ones((open_size, open_size), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(
        bright_surface,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    frame_area = float(width * height)
    ranked: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / max(frame_area, 1.0)
        if area_ratio < 0.035 or area_ratio > 0.75:
            continue
        quad = _quad_from_contour(contour)
        x, y, box_width, box_height = cv2.boundingRect(quad.astype(np.int32))
        if box_width < width * 0.18 or box_height < height * 0.16:
            continue
        aspect = box_width / max(float(box_height), 1.0)
        if not 0.75 <= aspect <= 3.2:
            continue
        quad_area = abs(float(cv2.contourArea(quad)))
        rectangularity = area / max(quad_area, 1.0)
        if rectangularity < 0.42:
            continue
        center_x = x + box_width * 0.5
        center_y = y + box_height * 0.5
        position_bonus = 1.0 + 0.18 * center_y / height + 0.08 * center_x / width
        ranked.append((area_ratio * min(rectangularity, 1.0) * position_bonus, quad))
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[0])[1]


def _rectify(frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
    source = _order_quad(quad)
    width = max(
        np.linalg.norm(source[1] - source[0]),
        np.linalg.norm(source[2] - source[3]),
    )
    height = max(
        np.linalg.norm(source[3] - source[0]),
        np.linalg.norm(source[2] - source[1]),
    )
    target_width = max(int(round(width)), 64)
    target_height = max(int(round(height)), 64)
    destination = np.float32(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        frame,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def neutralize_low_saturation(
    image: np.ndarray,
    saturation_threshold: int = 100,
) -> np.ndarray:
    """Remove tinted screen background while preserving saturated characters."""
    threshold = int(np.clip(saturation_threshold, 46, 220))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.where(hsv[:, :, 1] >= threshold, hsv[:, :, 1], 0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _place_on_model_canvas(image: np.ndarray) -> np.ndarray:
    max_width = int(round(CANVAS_WIDTH * DISPLAY_WIDTH_RATIO))
    max_height = int(round(CANVAS_HEIGHT * DISPLAY_HEIGHT_RATIO))
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 245, dtype=np.uint8)
    x0 = (CANVAS_WIDTH - resized_width) // 2
    y0 = int(round(CANVAS_HEIGHT * 0.17))
    y0 = min(max(y0, 0), CANVAS_HEIGHT - resized_height)
    canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized
    return canvas


def prepare_sign_frame(
    frame: np.ndarray,
    saturation_threshold: int = 100,
) -> np.ndarray:
    """Adapt a camera frame to the unchanged SignRecognizer input assumptions."""
    quad = find_bright_display_quad(frame)
    if quad is None:
        return neutralize_low_saturation(frame, saturation_threshold)
    rectified = _rectify(frame, quad)
    neutralized = neutralize_low_saturation(rectified, saturation_threshold)
    return _place_on_model_canvas(neutralized)

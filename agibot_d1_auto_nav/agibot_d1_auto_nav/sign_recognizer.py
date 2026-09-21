from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from agibot_d1_auto_nav.character_classifier import CharacterClassifier
from agibot_d1_auto_nav.color_classifier import build_chromatic_mask, classify_mask_color
from agibot_d1_auto_nav.vision_types import (
    CharacterCandidate,
    CharacterObservation,
    FrameObservation,
    SignResult,
)


@dataclass(frozen=True)
class VoteResult:
    value: str
    confidence: float
    agreement: float
    support: int


class SignRecognizer:
    """CPU-only sign recognizer for one digit and one capital letter.

    Pipeline:
      broad color segmentation -> geometry-based pair selection -> optional
      white-paper rectification -> HOG/PCA/KNN classification -> independent
      weighted multi-frame voting.
    """

    def __init__(
        self,
        min_valid_frames: int = 3,
        min_char_vote_ratio: float = 0.55,
        min_color_vote_ratio: float = 0.50,
        min_character_confidence: float = 0.35,
        min_color_confidence: float = 0.45,
        digit_position: str = 'right',
        classifier: CharacterClassifier | None = None,
    ) -> None:
        normalized_digit_position = str(digit_position).strip().lower()
        if normalized_digit_position not in {'left', 'right'}:
            raise ValueError(
                f'digit_position must be "left" or "right", got {digit_position!r}'
            )
        self.classifier = classifier if classifier is not None else CharacterClassifier()
        self.min_valid_frames = max(1, int(min_valid_frames))
        self.min_char_vote_ratio = float(min_char_vote_ratio)
        self.min_color_vote_ratio = float(min_color_vote_ratio)
        self.min_character_confidence = float(min_character_confidence)
        self.min_color_confidence = float(min_color_confidence)
        self.digit_position = normalized_digit_position

    def _assign_character_sides(self, left, right):
        if self.digit_position == 'left':
            return left, right
        return right, left

    @staticmethod
    def _component_mask(mask_roi: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_roi, 8)
        if count <= 1:
            return np.where(mask_roi > 0, 255, 0).astype(np.uint8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + int(np.argmax(areas))
        return np.where(labels == best_label, 255, 0).astype(np.uint8)

    @staticmethod
    def _candidate_pair_score(
        frame: np.ndarray,
        left: CharacterCandidate,
        right: CharacterCandidate,
    ) -> float:
        frame_height, frame_width = frame.shape[:2]
        lx, ly, lw, lh = left.bbox
        rx, ry, rw, rh = right.bbox
        average_height = (lh + rh) / 2.0

        height_ratio = min(lh, rh) / max(lh, rh)
        if height_ratio < 0.55:
            return -1e9

        vertical_offset = abs(left.center[1] - right.center[1]) / max(average_height, 1.0)
        if vertical_offset > 0.45:
            return -1e9

        horizontal_distance = (right.center[0] - left.center[0]) / max(average_height, 1.0)
        if horizontal_distance < 0.35 or horizontal_distance > 2.20:
            return -1e9

        pair_center_x = (left.center[0] + right.center[0]) / (2.0 * frame_width)
        pair_center_y = (left.center[1] + right.center[1]) / (2.0 * frame_height)

        search_x0 = max(0, int(lx - 0.35 * average_height))
        search_y0 = max(0, int(min(ly, ry) - 0.45 * average_height))
        search_x1 = min(frame_width, int(rx + rw + 0.35 * average_height))
        search_y1 = min(frame_height, int(max(ly + lh, ry + rh) + 0.65 * average_height))
        support_patch = frame[search_y0:search_y1, search_x0:search_x1]
        if support_patch.size:
            support_hsv = cv2.cvtColor(support_patch, cv2.COLOR_BGR2HSV)
            white_support = float(np.mean(
                (support_hsv[:, :, 1] < 75) & (support_hsv[:, :, 2] > 115)
            ))
        else:
            white_support = 0.0

        combined_area = left.contour_area + right.contour_area
        return float(
            2.5 * height_ratio
            - 2.5 * vertical_offset
            - 1.2 * abs(horizontal_distance - 1.05)
            - 1.5 * abs(pair_center_x - 0.50)
            + 1.2 * pair_center_y
            + 2.0 * white_support
            + 0.3 * math.log1p(combined_area)
        )

    def _detect_candidates(self, frame: np.ndarray) -> list[CharacterCandidate]:
        frame_height, frame_width = frame.shape[:2]
        hsv, broad_mask = build_chromatic_mask(frame)
        broad_mask[:int(frame_height * 0.18), :] = 0
        contours, _ = cv2.findContours(
            broad_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[CharacterCandidate] = []
        minimum_area = max(180.0, frame_height * frame_width * 0.00008)
        maximum_area = frame_height * frame_width * 0.08

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < minimum_area or area > maximum_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if height < frame_height * 0.07 or height > frame_height * 0.65:
                continue
            aspect_ratio = width / max(float(height), 1.0)
            if aspect_ratio < 0.12 or aspect_ratio > 1.65:
                continue
            fill_ratio = area / max(width * height, 1)
            if fill_ratio < 0.08:
                continue

            padding = max(4, int(round(0.035 * max(width, height))))
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(frame_width, x + width + padding)
            y1 = min(frame_height, y + height + padding)
            local_mask = broad_mask[y0:y1, x0:x1]
            component_mask = self._component_mask(local_mask)
            if cv2.countNonZero(component_mask) < 80:
                continue
            local_hsv = hsv[y0:y1, x0:x1]
            color = classify_mask_color(local_hsv, component_mask)
            candidates.append(
                CharacterCandidate(
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    mask=component_mask,
                    center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                    contour_area=area,
                    fill_ratio=fill_ratio,
                    color=color,
                )
            )
        return candidates

    def _select_pair(
        self,
        frame: np.ndarray,
        candidates: list[CharacterCandidate],
    ) -> tuple[CharacterCandidate, CharacterCandidate, float]:
        best: tuple[CharacterCandidate, CharacterCandidate, float] | None = None
        for first, second in itertools.combinations(candidates, 2):
            left, right = (first, second) if first.center[0] < second.center[0] else (second, first)
            score = self._candidate_pair_score(frame, left, right)
            if best is None or score > best[2]:
                best = (left, right, score)
        if best is None or best[2] < 4.5:
            raise RuntimeError('no geometrically valid digit-letter pair')
        return best

    @staticmethod
    def _order_quad(points: np.ndarray) -> np.ndarray:
        points = points.astype(np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)
        point_sum = points.sum(axis=1)
        point_diff = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[np.argmin(point_sum)]
        ordered[2] = points[np.argmax(point_sum)]
        ordered[1] = points[np.argmin(point_diff)]
        ordered[3] = points[np.argmax(point_diff)]
        return ordered

    def _localize_white_sign(
        self,
        frame: np.ndarray,
        left: CharacterCandidate,
        right: CharacterCandidate,
    ) -> np.ndarray | None:
        frame_height, frame_width = frame.shape[:2]
        lx, ly, lw, lh = left.bbox
        rx, ry, rw, rh = right.bbox
        average_height = (lh + rh) / 2.0
        x0 = max(0, int(lx - 0.90 * average_height))
        y0 = max(0, int(min(ly, ry) - 1.05 * average_height))
        x1 = min(frame_width, int(rx + rw + 0.90 * average_height))
        y1 = min(frame_height, int(max(ly + lh, ry + rh) + 1.75 * average_height))
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        white_mask = (
            (hsv[:, :, 1] < 80)
            & (hsv[:, :, 2] > 105)
        ).astype(np.uint8) * 255
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            np.ones((17, 17), dtype=np.uint8),
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            np.ones((5, 5), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        left_local = (left.center[0] - x0, left.center[1] - y0)
        right_local = (right.center[0] - x0, right.center[1] - y0)
        ranked: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < average_height * average_height * 1.4:
                continue
            inside_left = cv2.pointPolygonTest(contour, left_local, False) >= 0
            inside_right = cv2.pointPolygonTest(contour, right_local, False) >= 0
            if not inside_left or not inside_right:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                local_quad = approx.reshape(4, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(contour)
                local_quad = cv2.boxPoints(rect).astype(np.float32)
            score = float(area)
            ranked.append((score, local_quad))

        if not ranked:
            return None
        local_quad = max(ranked, key=lambda item: item[0])[1]
        local_quad[:, 0] += x0
        local_quad[:, 1] += y0
        return self._order_quad(local_quad)

    @staticmethod
    def _warp_sign(frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        destination = np.array(
            [[0, 0], [719, 0], [719, 899], [0, 899]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
        return cv2.warpPerspective(frame, matrix, (720, 900), borderValue=(255, 255, 255))

    def recognize_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameObservation:
        initial_candidates = self._detect_candidates(frame)
        left, right, pair_score = self._select_pair(frame, initial_candidates)
        sign_quad = self._localize_white_sign(frame, left, right)
        digit_candidate, letter_candidate = self._assign_character_sides(left, right)

        digit_prediction = self.classifier.predict_digit(digit_candidate.mask)
        letter_prediction = self.classifier.predict_letter(letter_candidate.mask)

        # Perspective rectification is an optional enhancement, not a mandatory
        # dependency. Keep the original pair unless the rectified view produces
        # a clearly stronger joint classification score. This prevents an
        # imperfect white-paper contour from degrading an otherwise clean frame.
        if sign_quad is not None:
            try:
                rectified = self._warp_sign(frame, sign_quad)
                rectified_candidates = self._detect_candidates(rectified)
                rect_left, rect_right, rect_score = self._select_pair(rectified, rectified_candidates)
                rect_digit_candidate, rect_letter_candidate = self._assign_character_sides(
                    rect_left,
                    rect_right,
                )
                rect_digit = self.classifier.predict_digit(rect_digit_candidate.mask)
                rect_letter = self.classifier.predict_letter(rect_letter_candidate.mask)
                original_joint = min(
                    digit_prediction.confidence,
                    letter_prediction.confidence,
                )
                rectified_joint = min(rect_digit.confidence, rect_letter.confidence)
                if rect_score >= 4.5 and rectified_joint > original_joint + 0.08:
                    digit_prediction = rect_digit
                    letter_prediction = rect_letter
            except Exception:
                pass
        return FrameObservation(
            frame_index=frame_index,
            frame=frame,
            digit=CharacterObservation(digit_candidate, digit_prediction),
            letter=CharacterObservation(letter_candidate, letter_prediction),
            pair_score=pair_score,
            sign_quad=sign_quad,
        )

    @staticmethod
    def _weighted_vote(
        values: list[str],
        weights: list[float],
        confidences: list[float],
    ) -> VoteResult:
        if not values:
            raise RuntimeError('no values available for voting')
        totals: dict[str, float] = {}
        counts: Counter[str] = Counter()
        confidence_totals: dict[str, float] = {}
        for value, weight, confidence in zip(values, weights, confidences):
            totals[value] = totals.get(value, 0.0) + max(weight, 1e-4)
            counts[value] += 1
            confidence_totals[value] = confidence_totals.get(value, 0.0) + confidence
        best_value = max(totals, key=lambda value: (totals[value], counts[value]))
        total_weight = sum(totals.values())
        agreement = totals[best_value] / max(total_weight, 1e-6)
        mean_confidence = confidence_totals[best_value] / max(counts[best_value], 1)
        confidence = float(np.clip(0.65 * agreement + 0.35 * mean_confidence, 0.0, 1.0))
        return VoteResult(best_value, confidence, agreement, counts[best_value])

    def aggregate(self, observations: list[FrameObservation]) -> tuple[SignResult, FrameObservation]:
        if len(observations) < self.min_valid_frames:
            raise RuntimeError(
                f'only {len(observations)} valid frames, require {self.min_valid_frames}'
            )

        weights = [max(item.quality, 0.05) for item in observations]
        digit_vote = self._weighted_vote(
            [item.digit.prediction.char for item in observations],
            weights,
            [item.digit.prediction.confidence for item in observations],
        )
        letter_vote = self._weighted_vote(
            [item.letter.prediction.char for item in observations],
            weights,
            [item.letter.prediction.confidence for item in observations],
        )
        digit_color_vote = self._weighted_vote(
            [item.digit.candidate.color.name for item in observations],
            weights,
            [item.digit.candidate.color.confidence for item in observations],
        )
        letter_color_vote = self._weighted_vote(
            [item.letter.candidate.color.name for item in observations],
            weights,
            [item.letter.candidate.color.confidence for item in observations],
        )

        if digit_vote.agreement < self.min_char_vote_ratio:
            raise RuntimeError(f'digit vote agreement too low: {digit_vote.agreement:.2f}')
        if letter_vote.agreement < self.min_char_vote_ratio:
            raise RuntimeError(f'letter vote agreement too low: {letter_vote.agreement:.2f}')
        if digit_color_vote.agreement < self.min_color_vote_ratio:
            raise RuntimeError(
                f'digit color vote agreement too low: {digit_color_vote.agreement:.2f}'
            )
        if letter_color_vote.agreement < self.min_color_vote_ratio:
            raise RuntimeError(
                f'letter color vote agreement too low: {letter_color_vote.agreement:.2f}'
            )
        if digit_color_vote.value == 'unknown' or letter_color_vote.value == 'unknown':
            raise RuntimeError('color vote resolved to unknown')

        matching = [
            item for item in observations
            if item.digit.prediction.char == digit_vote.value
            and item.letter.prediction.char == letter_vote.value
        ]
        representative = max(matching or observations, key=lambda item: item.quality)
        result = SignResult(
            number=digit_vote.value,
            number_color=digit_color_vote.value,
            letter=letter_vote.value,
            letter_color=letter_color_vote.value,
            image_path=Path(),
            annotated_path=Path(),
            number_confidence=digit_vote.confidence,
            number_color_confidence=digit_color_vote.confidence,
            letter_confidence=letter_vote.confidence,
            letter_color_confidence=letter_color_vote.confidence,
            valid_frame_count=len(observations),
            total_frame_count=len(observations),
        )
        return result, representative

    @staticmethod
    def _draw_observation(
        observation: FrameObservation,
        result: SignResult,
        output_path: Path,
    ) -> None:
        annotated = observation.frame.copy()
        if observation.sign_quad is not None:
            quad = observation.sign_quad.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [quad], True, (220, 220, 220), 3)

        items = [
            ('digit', observation.digit, result.number, result.number_color),
            ('letter', observation.letter, result.letter, result.letter_color),
        ]
        for role, item, voted_char, voted_color in items:
            x, y, width, height = item.candidate.bbox
            bgr = item.candidate.color.bgr
            cv2.rectangle(annotated, (x, y), (x + width, y + height), bgr, 4)
            label = (
                f'{voted_char}/{voted_color} '
                f'C{item.prediction.confidence:.2f} '
                f'K{item.candidate.color.confidence:.2f}'
            )
            label_origin = (x, max(y - 10, 24))
            text_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )
            cv2.rectangle(
                annotated,
                (label_origin[0] - 2, label_origin[1] - text_size[1] - 4),
                (label_origin[0] + text_size[0] + 3, label_origin[1] + baseline + 2),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                label_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                bgr,
                2,
                cv2.LINE_AA,
            )

        lines = [
            f'number: {result.number} ({result.number_confidence:.2f})',
            f'number_color: {result.number_color} ({result.number_color_confidence:.2f})',
            f'letter: {result.letter} ({result.letter_confidence:.2f})',
            f'letter_color: {result.letter_color} ({result.letter_color_confidence:.2f})',
            f'valid_frames: {result.valid_frame_count}/{result.total_frame_count}',
        ]
        overlay_height = 180
        canvas = np.full(
            (annotated.shape[0] + overlay_height, annotated.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        canvas[:annotated.shape[0]] = annotated
        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (28, annotated.shape[0] + 34 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f'failed to save annotated image: {output_path}')

    def recognize_frames(
        self,
        frames: list[np.ndarray],
        output_dir: Path,
        save_all_frames: bool = False,
    ) -> SignResult:
        observations: list[FrameObservation] = []
        errors: list[str] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, frame in enumerate(frames):
            try:
                observation = self.recognize_frame(frame, frame_index=index)
                if observation.digit.prediction.confidence < self.min_character_confidence:
                    raise RuntimeError(
                        f'low digit confidence {observation.digit.prediction.confidence:.2f}'
                    )
                if observation.letter.prediction.confidence < self.min_character_confidence:
                    raise RuntimeError(
                        f'low letter confidence {observation.letter.prediction.confidence:.2f}'
                    )
                if observation.digit.candidate.color.confidence < self.min_color_confidence:
                    raise RuntimeError(
                        f'low digit color confidence {observation.digit.candidate.color.confidence:.2f}'
                    )
                if observation.letter.candidate.color.confidence < self.min_color_confidence:
                    raise RuntimeError(
                        f'low letter color confidence {observation.letter.candidate.color.confidence:.2f}'
                    )
                observations.append(observation)
                if save_all_frames:
                    path = output_dir / f'accepted_{index + 1:02d}.jpg'
                    provisional = SignResult(
                        number=observation.digit.prediction.char,
                        number_color=observation.digit.candidate.color.name,
                        letter=observation.letter.prediction.char,
                        letter_color=observation.letter.candidate.color.name,
                        image_path=path,
                        annotated_path=path,
                    )
                    self._draw_observation(observation, provisional, path)
            except Exception as exc:
                errors.append(f'frame {index}: {exc}')

        result, representative = self.aggregate(observations)
        result.total_frame_count = len(frames)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_path = output_dir / f'sign_capture_{timestamp}.jpg'
        annotated_path = output_dir / f'sign_result_{timestamp}.jpg'
        if not cv2.imwrite(str(image_path), representative.frame):
            raise RuntimeError(f'failed to save source image: {image_path}')
        result.image_path = image_path
        result.annotated_path = annotated_path
        self._draw_observation(representative, result, annotated_path)

        if errors:
            error_log = output_dir / f'sign_errors_{timestamp}.txt'
            error_log.write_text('\n'.join(errors), encoding='utf-8')
        return result

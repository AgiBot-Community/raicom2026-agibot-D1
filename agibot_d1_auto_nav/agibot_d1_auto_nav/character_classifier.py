from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from agibot_d1_auto_nav.vision_types import CharacterPrediction


IMAGE_SIZE = 96
DIGITS = '0123456789'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'synthetic_hog_pca_knn.npz'


class CharacterClassifier:
    """Small CPU classifier: HOG -> PCA -> KNN.

    The model data is pre-generated and shipped with the ROS package. Runtime
    dependencies remain NumPy and OpenCV only.
    """

    def __init__(self, model_path: Path = MODEL_PATH, k: int = 9) -> None:
        if not model_path.is_file():
            raise RuntimeError(f'character model file not found: {model_path}')
        data = np.load(str(model_path), allow_pickle=False)
        self._mean = data['pca_mean'].astype(np.float32)
        self._components = data['pca_components'].astype(np.float32)
        self._digit_chars = str(data['digit_chars'].item())
        self._letter_chars = str(data['letter_chars'].item())
        self._k = int(k)

        self._digit_knn = cv2.ml.KNearest_create()
        self._digit_knn.train(
            data['digit_samples'].astype(np.float32),
            cv2.ml.ROW_SAMPLE,
            data['digit_labels'].astype(np.float32),
        )
        self._letter_knn = cv2.ml.KNearest_create()
        self._letter_knn.train(
            data['letter_samples'].astype(np.float32),
            cv2.ml.ROW_SAMPLE,
            data['letter_labels'].astype(np.float32),
        )
        self._hog = cv2.HOGDescriptor(
            _winSize=(IMAGE_SIZE, IMAGE_SIZE),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

    @staticmethod
    def normalize_mask(binary_mask: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
        mask = np.where(binary_mask > 0, 255, 0).astype(np.uint8)
        coordinates = cv2.findNonZero(mask)
        if coordinates is None:
            return np.zeros((size, size), dtype=np.uint8)

        x, y, width, height = cv2.boundingRect(coordinates)
        crop = mask[y:y + height, x:x + width]
        target = size - 16
        scale = min(target / max(width, 1), target / max(height, 1))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        normalized = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - resized_width) // 2
        y_offset = (size - resized_height) // 2
        normalized[
            y_offset:y_offset + resized_height,
            x_offset:x_offset + resized_width,
        ] = resized
        _, normalized = cv2.threshold(normalized, 100, 255, cv2.THRESH_BINARY)
        return normalized

    def _project(self, normalized: np.ndarray) -> np.ndarray:
        feature = self._hog.compute(normalized).reshape(1, -1).astype(np.float32)
        return (feature - self._mean) @ self._components.T

    @staticmethod
    def _decode_vote(
        chars: str,
        neighbors: np.ndarray,
        distances: np.ndarray,
    ) -> CharacterPrediction:
        labels = neighbors.reshape(-1).astype(np.int32)
        distances = np.sqrt(np.maximum(distances.reshape(-1), 0.0))
        weights = 1.0 / (distances + 1e-3)
        class_weights: dict[int, float] = {}
        class_counts: dict[int, int] = {}
        for label, weight in zip(labels, weights):
            class_weights[int(label)] = class_weights.get(int(label), 0.0) + float(weight)
            class_counts[int(label)] = class_counts.get(int(label), 0) + 1

        ranked = sorted(class_weights.items(), key=lambda item: item[1], reverse=True)
        best_label, best_weight = ranked[0]
        second_weight = ranked[1][1] if len(ranked) > 1 else 0.0
        total_weight = sum(class_weights.values())
        vote_ratio = best_weight / max(total_weight, 1e-6)
        count_ratio = class_counts[best_label] / max(len(labels), 1)
        margin = (best_weight - second_weight) / max(best_weight, 1e-6)
        neighbor_quality = float(np.exp(-0.08 * float(np.median(distances))))
        relative_confidence = (
            0.45 * vote_ratio + 0.35 * count_ratio + 0.20 * margin
        )
        confidence = float(np.clip(
            relative_confidence * neighbor_quality,
            0.0,
            1.0,
        ))
        return CharacterPrediction(chars[best_label], confidence, vote_ratio, margin)

    def predict_digit(self, binary_mask: np.ndarray) -> CharacterPrediction:
        normalized = self.normalize_mask(binary_mask)
        projected = self._project(normalized)
        _, _, neighbors, distances = self._digit_knn.findNearest(projected, k=self._k)
        return self._decode_vote(self._digit_chars, neighbors, distances)

    def predict_letter(self, binary_mask: np.ndarray) -> CharacterPrediction:
        normalized = self.normalize_mask(binary_mask)
        projected = self._project(normalized)
        _, _, neighbors, distances = self._letter_knn.findNearest(projected, k=self._k)
        return self._decode_vote(self._letter_chars, neighbors, distances)

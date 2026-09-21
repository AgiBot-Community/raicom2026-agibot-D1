from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ColorPrediction:
    name: str
    confidence: float
    dominance: float
    pixel_count: int
    bgr: tuple[int, int, int]


@dataclass(frozen=True)
class CharacterPrediction:
    char: str
    confidence: float
    vote_ratio: float
    margin: float


@dataclass
class CharacterCandidate:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    center: tuple[float, float]
    contour_area: float
    fill_ratio: float
    color: ColorPrediction


@dataclass
class CharacterObservation:
    candidate: CharacterCandidate
    prediction: CharacterPrediction


@dataclass
class FrameObservation:
    frame_index: int
    frame: np.ndarray
    digit: CharacterObservation
    letter: CharacterObservation
    pair_score: float
    sign_quad: np.ndarray | None

    @property
    def quality(self) -> float:
        char_quality = min(
            self.digit.prediction.confidence,
            self.letter.prediction.confidence,
        )
        color_quality = min(
            self.digit.candidate.color.confidence,
            self.letter.candidate.color.confidence,
        )
        pair_quality = max(0.0, min(1.0, (self.pair_score - 4.0) / 4.0))
        return 0.55 * char_quality + 0.25 * color_quality + 0.20 * pair_quality


@dataclass
class SignResult:
    number: str
    number_color: str
    letter: str
    letter_color: str
    image_path: Path
    annotated_path: Path
    number_confidence: float = 0.0
    number_color_confidence: float = 0.0
    letter_confidence: float = 0.0
    letter_color_confidence: float = 0.0
    valid_frame_count: int = 0
    total_frame_count: int = 0

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA


IMAGE_SIZE = 96
DIGITS = '0123456789'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
    '/usr/share/fonts/truetype/lato/Lato-Medium.ttf',
    '/usr/share/fonts/truetype/lato/Lato-Bold.ttf',
    '/usr/share/fonts/truetype/lato/Lato-Heavy.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/opentype/freefont/FreeSans.otf',
    '/usr/share/opentype/freefont/FreeSansBold.otf',
    '/usr/share/opentype/inter/InterDisplay-Regular.otf',
    '/usr/share/opentype/inter/InterDisplay-Light.otf',
    '/usr/share/opentype/inter/InterDisplay-Bold.otf',
]


def normalize(mask: np.ndarray) -> np.ndarray:
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    x, y, width, height = cv2.boundingRect(coordinates)
    crop = mask[y:y + height, x:x + width]
    target = IMAGE_SIZE - 16
    scale = min(target / width, target / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_AREA)
    output = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    x_offset = (IMAGE_SIZE - new_width) // 2
    y_offset = (IMAGE_SIZE - new_height) // 2
    output[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    _, output = cv2.threshold(output, 100, 255, cv2.THRESH_BINARY)
    return output


def render_character(
    char: str,
    font_path: Path,
    font_size: int,
    angle: float,
    horizontal_scale: float,
) -> np.ndarray:
    canvas = Image.new('L', (160, 160), 0)
    font = ImageFont.truetype(str(font_path), font_size)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), char, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    position = ((160 - width) // 2 - box[0], (160 - height) // 2 - box[1])
    draw.text(position, char, fill=255, font=font)
    image = np.asarray(canvas)

    scaled_width = max(1, int(round(image.shape[1] * horizontal_scale)))
    image = cv2.resize(image, (scaled_width, image.shape[0]), interpolation=cv2.INTER_LINEAR)
    if scaled_width >= 160:
        start = (scaled_width - 160) // 2
        image = image[:, start:start + 160]
    else:
        padded = np.zeros((160, 160), dtype=np.uint8)
        start = (160 - scaled_width) // 2
        padded[:, start:start + scaled_width] = image
        image = padded

    transform = cv2.getRotationMatrix2D((80, 80), angle, 1.0)
    image = cv2.warpAffine(image, transform, (160, 160), borderValue=0)
    _, image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    return normalize(image)


def build_features(chars: str, hog: cv2.HOGDescriptor, fonts: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    labels: list[int] = []
    for label, char in enumerate(chars):
        for font_path in fonts:
            for font_size in (88, 108):
                for angle in (-8.0, 0.0, 8.0):
                    for horizontal_scale in (0.90, 1.0, 1.10):
                        image = render_character(
                            char,
                            font_path,
                            font_size,
                            angle,
                            horizontal_scale,
                        )
                        samples.append(hog.compute(image).reshape(-1))
                        labels.append(label)
    return np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--components', type=int, default=192)
    args = parser.parse_args()

    fonts = [Path(path) for path in FONT_CANDIDATES if Path(path).is_file()]
    if len(fonts) < 4:
        raise RuntimeError('not enough fonts found to build character model')

    hog = cv2.HOGDescriptor(
        _winSize=(IMAGE_SIZE, IMAGE_SIZE),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    digit_features, digit_labels = build_features(DIGITS, hog, fonts)
    letter_features, letter_labels = build_features(LETTERS, hog, fonts)
    all_features = np.vstack([digit_features, letter_features])

    pca = PCA(n_components=args.components, svd_solver='randomized', random_state=7)
    all_projected = pca.fit_transform(all_features).astype(np.float32)
    digit_count = len(digit_features)
    digit_projected = all_projected[:digit_count]
    letter_projected = all_projected[digit_count:]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pca_mean=pca.mean_.reshape(1, -1).astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        digit_samples=digit_projected,
        digit_labels=digit_labels,
        letter_samples=letter_projected,
        letter_labels=letter_labels,
        digit_chars=np.asarray(DIGITS),
        letter_chars=np.asarray(LETTERS),
    )
    print(f'wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)')


if __name__ == '__main__':
    main()

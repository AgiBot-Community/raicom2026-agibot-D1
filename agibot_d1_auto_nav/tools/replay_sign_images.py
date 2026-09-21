#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from agibot_d1_auto_nav.sign_recognizer import SignRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline replay for sign images.')
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--pattern', default='*.jpg')
    parser.add_argument('--output-dir', type=Path, default=Path('vision_replay_output'))
    parser.add_argument('--min-valid-frames', type=int, default=3)
    args = parser.parse_args()

    image_paths = sorted(args.input_dir.glob(args.pattern))
    if not image_paths:
        raise SystemExit(f'no images matched {args.input_dir / args.pattern}')

    recognizer = SignRecognizer(min_valid_frames=args.min_valid_frames)
    observations = []
    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append({'image': str(image_path), 'status': 'read_failed'})
            continue
        try:
            observation = recognizer.recognize_frame(image, frame_index=index)
            observations.append(observation)
            rows.append({
                'image': str(image_path),
                'status': 'ok',
                'number': observation.digit.prediction.char,
                'number_confidence': f'{observation.digit.prediction.confidence:.4f}',
                'number_color': observation.digit.candidate.color.name,
                'number_color_confidence': f'{observation.digit.candidate.color.confidence:.4f}',
                'letter': observation.letter.prediction.char,
                'letter_confidence': f'{observation.letter.prediction.confidence:.4f}',
                'letter_color': observation.letter.candidate.color.name,
                'letter_color_confidence': f'{observation.letter.candidate.color.confidence:.4f}',
                'pair_score': f'{observation.pair_score:.4f}',
            })
        except Exception as exc:
            rows.append({
                'image': str(image_path),
                'status': 'failed',
                'error': str(exc),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'frame_results.csv'
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open('w', newline='', encoding='utf-8-sig') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result, representative = recognizer.aggregate(observations)
    result.total_frame_count = len(image_paths)
    result.image_path = args.output_dir / 'representative.jpg'
    result.annotated_path = args.output_dir / 'result.jpg'
    cv2.imwrite(str(result.image_path), representative.frame)
    recognizer._draw_observation(representative, result, result.annotated_path)

    print(
        f'{result.number}/{result.number_color} '
        f'{result.letter}/{result.letter_color} '
        f'valid={result.valid_frame_count}/{result.total_frame_count}'
    )
    print(f'CSV: {csv_path}')
    print(f'Annotated: {result.annotated_path}')


if __name__ == '__main__':
    main()
